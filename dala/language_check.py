"""Local LanguageTool screening with versioned, persistent response caching."""
from contextlib import contextmanager, ExitStack
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import time
import threading

import requests

# Portable Linux runtime used by the documented reproducible build.
JAVA_URL = ('https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.16%2B8/'
            'OpenJDK17U-jre_x64_linux_hotspot_17.0.16_8.tar.gz')
LT_URL = 'https://languagetool.org/download/LanguageTool-6.6.zip'
HARD_TYPES = {'grammar', 'misspelling', 'typographical'}


def prepare_runtime(directory='la_output/tools'):
    """Explicit setup command downloads pinned local tools; never uses a cloud checker."""
    import tarfile
    import zipfile
    from .common_pile import sha256_file
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, url in [('java.tar.gz', JAVA_URL), ('languagetool.zip', LT_URL)]:
        target = root / name
        if not target.exists():
            temporary = target.with_suffix('.part')
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            temporary.replace(target)
        receipts[name] = dict(url=url, sha256=sha256_file(target))
    receipt_path = root / 'runtime.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipts:
        raise ValueError('Local tool archives differ from their recorded checksums')
    if not (root / 'jdk-17.0.16+8-jre/bin/java').exists():
        with tarfile.open(root / 'java.tar.gz') as archive:
            archive.extractall(root, filter='data')
    if not (root / 'LanguageTool-6.6/languagetool-server.jar').exists():
        with zipfile.ZipFile(root / 'languagetool.zip') as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root.resolve()):
                    raise ValueError('Unsafe archive member')
            archive.extractall(root)
    receipt_path.write_text(json.dumps(receipts, indent=2) + '\n')
    return receipts


@contextmanager
def local_server(directory='la_output/tools', threads=8, heap='2g', instance=None):
    root = Path(directory).resolve()
    java = root / 'jdk-17.0.16+8-jre/bin/java'
    server = root / 'LanguageTool-6.6/languagetool-server.jar'
    if not java.exists() or not server.exists():
        raise RuntimeError('Install local checker with: python -m dala.language_check --setup')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    stem = 'server' if instance is None else f'server-{instance}-{port}'
    config = root / f'{stem}.properties'
    config.write_text(f'cacheSize=10000\nmaxCheckThreads={int(threads)}\n')
    log_path = root / f'{stem}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([str(java), f'-Xmx{heap}', '-cp', str(server),
            'org.languagetool.server.HTTPServer', '--config', str(config), '--port', str(port)],
            stdout=log, stderr=subprocess.STDOUT, cwd=server.parent)
        url = f'http://127.0.0.1:{port}'
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'Local LanguageTool failed; inspect {log_path}')
                try:
                    r = requests.get(url + '/v2/languages', timeout=1)
                    if r.ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError('Local LanguageTool did not become ready')
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


@contextmanager
def local_servers(directory='la_output/tools', instances=1, threads=8, heap='2g'):
    """Identical local checker processes; rules and diagnostic handling unchanged."""
    if not isinstance(instances, int) or instances < 1:
        raise ValueError('Checker instances must be a positive integer')
    with ExitStack() as stack:
        urls = [stack.enter_context(local_server(directory, threads, heap,
                    instance=i if instances > 1 else None)) for i in range(instances)]
        yield urls


class SentenceCheckFailure(RuntimeError):
    """An unverified sentence must be rejected, never treated as checker success."""


class CheckerUnavailable(RuntimeError):
    """Health or failure-budget guard stopped screening."""


class IncompleteCheck(RuntimeError):
    pass


class LanguageCheck:
    def __init__(self, url, cache='la_output/cache/languagetool.sqlite3', dialects=None, commit_batch_size=1, lexical_carrier='The word is ', hard_rule_ids=(), failure_policy=None):
        self.failure_policy = failure_policy or {}
        if self.failure_policy:
            if self.failure_policy.get('mode') != 'reject_sentence' or not self.failure_policy.get('log_path'):
                raise ValueError('Checker resilience requires reject_sentence mode and an audit log')
            for name, default in [('attempts',3),('window',1000),('max_failures',20),('max_consecutive',5)]:
                value=self.failure_policy.get(name,default)
                if not isinstance(value,int) or value<1:raise ValueError('Positive failure-policy limits required')
            if self.failure_policy.get('max_failures',20)>self.failure_policy.get('window',1000):
                raise ValueError('Failure budget exceeds its window')
            Path(self.failure_policy['log_path']).parent.mkdir(parents=True,exist_ok=True)
        self.failure_outcomes=deque(maxlen=self.failure_policy.get('window',1000))
        self.consecutive_failures=0
        self.circuit_open=False
        self.dialects = dialects or ('en-US', 'en-GB')
        self.lexical_carrier = lexical_carrier
        self.hard_rule_ids = frozenset(hard_rule_ids)
        self.urls = tuple(u.rstrip('/') for u in ([url] if isinstance(url, str) else url))
        if not self.urls or any(not u.startswith(('http://127.0.0.1:', 'http://localhost:')) for u in self.urls):
            raise ValueError('Only a local LanguageTool server is supported')
        self.url = self.urls[0]
        self.next_server = 0
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache, check_same_thread=False)
        self.lock = threading.Lock()
        self.commit_batch_size = max(1, int(commit_batch_size))
        self.pending_commits = 0
        self.local = threading.local()
        self.sessions = []
        self.db.execute('CREATE TABLE IF NOT EXISTS checks (key TEXT PRIMARY KEY, response TEXT NOT NULL)')
        self.session = requests.Session()
        self.sessions.append(self.session)
        probe = self.session.post(self.url + '/v2/check', data={'language': self.dialects[0], 'text': 'This is a sentence.'}, timeout=60)
        probe.raise_for_status()
        self.software = probe.json()['software']
        for endpoint in self.urls[1:]:
            other = self.session.post(endpoint + '/v2/check', data={'language': self.dialects[0], 'text': 'This is a sentence.'}, timeout=60)
            other.raise_for_status()
            if other.json()['software'] != self.software:
                self.close()
                raise ValueError('Checker pool software versions differ')
        self.namespace = json.dumps(self.software, sort_keys=True)

    def check(self, text, language):
        key = hashlib.sha256(f'{self.namespace}\0{language}\0{text}'.encode()).hexdigest()
        with self.lock:
            cached = self.db.execute('SELECT response FROM checks WHERE key=?', (key,)).fetchone()
        if cached:
            return json.loads(cached[0])
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            with self.lock:
                self.sessions.append(self.local.session)
                self.local.url = self.urls[self.next_server % len(self.urls)]
                self.next_server += 1
        if self.failure_policy:
            data = self._resilient_check(text, language)
        else:
            for attempt in range(4):
                try:
                    response = self.local.session.post(self.local.url + '/v2/check', data={'language': language, 'text': text}, timeout=60)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    self.local.session.close()
                    if attempt == 3:raise
                    time.sleep(.25 * 2 ** attempt)
            response.raise_for_status()
            data = response.json()
            if data.get('warnings', {}).get('incompleteResults'):
                raise RuntimeError('LanguageTool returned incomplete results')
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO checks VALUES (?,?)', (key, json.dumps(data)))
            self.pending_commits += 1
            if self.pending_commits >= self.commit_batch_size:
                self.db.commit()
                self.pending_commits = 0
        return data

    def _complete_response(self, response):
        response.raise_for_status()
        try:data=response.json()
        except ValueError as exc:raise IncompleteCheck('Invalid checker JSON') from exc
        if not isinstance(data,dict) or not isinstance(data.get('matches'),list):
            raise IncompleteCheck('Invalid checker response structure')
        if data.get('warnings',{}).get('incompleteResults'):
            raise IncompleteCheck('LanguageTool returned incomplete results')
        if data.get('software') != self.software:
            raise CheckerUnavailable('Checker software identity changed')
        return data

    def _failure_event(self, text, language, attempts, health_ok):
        event=dict(at=datetime.now(timezone.utc).isoformat(),language=language,
                   source_sha256=hashlib.sha256(text.encode()).hexdigest(),original=text,
                   checker_software=self.software,attempts=attempts,health_ok=health_ok,
                   disposition='sentence_rejected' if health_ok else 'checker_unavailable')
        with self.lock:
            if health_ok:
                self.failure_outcomes.append(True);self.consecutive_failures+=1
                if (sum(self.failure_outcomes)>=self.failure_policy.get('max_failures',20)
                        or self.consecutive_failures>=self.failure_policy.get('max_consecutive',5)):
                    self.circuit_open=True
                    event['disposition']='failure_budget_exceeded'
            else:self.circuit_open=True
            with Path(self.failure_policy['log_path']).open('a') as stream:
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
            stopped=self.circuit_open
        if stopped:raise CheckerUnavailable(event['disposition'])
        raise SentenceCheckFailure('Unverified sentence rejected; failure recorded')

    def _resilient_check(self, text, language):
        with self.lock:
            if self.circuit_open:raise CheckerUnavailable('Checker failure circuit is open')
        start=self.urls.index(self.local.url);attempts=[]
        for attempt in range(self.failure_policy.get('attempts',3)):
            endpoint=self.urls[(start+attempt)%len(self.urls)]
            try:
                response=self.local.session.post(endpoint+'/v2/check',data={'language':language,'text':text},timeout=60)
                data=self._complete_response(response)
                with self.lock:
                    if self.circuit_open:raise CheckerUnavailable('Checker failure circuit is open')
                    self.failure_outcomes.append(False);self.consecutive_failures=0
                self.local.url=endpoint
                return data
            except (requests.ConnectionError,requests.Timeout,requests.HTTPError,IncompleteCheck) as exc:
                failed=getattr(exc,'response',None)
                status=failed.status_code if failed is not None else None
                if status is not None and status<500 and status not in {408,429}:raise
                attempts.append(dict(endpoint=endpoint,error=type(exc).__name__,status=status,
                                     detail=(failed.text if failed is not None else str(exc))[:2000]))
                self.local.session.close()
                if attempt+1<self.failure_policy.get('attempts',3):time.sleep(.25*2**attempt)
        # An uncached control distinguishes sentence-specific failures from outages.
        health_ok=False
        for endpoint in dict.fromkeys(a['endpoint'] for a in attempts):
            try:
                control=self.local.session.post(endpoint+'/v2/check',data={'language':language,
                    'text':self.failure_policy.get('health_text','This is a sentence.')},timeout=15)
                self._complete_response(control);health_ok=True;break
            except (requests.RequestException,IncompleteCheck):continue
        self._failure_event(text,language,attempts,health_ok)

    @staticmethod
    def hard_matches(data, text, hard_rule_ids=()):
        """Convert Java UTF-16 offsets to Python character offsets."""
        out = []
        utf16 = text.encode('utf-16-le')
        for match in data['matches']:
            rule = match['rule']
            issue_type = rule.get('issueType')
            # Some Dutch agreement rules are labeled as spelling or uncategorized.
            # Only explicitly reviewed rule IDs may override their category.
            if issue_type in {'uncategorized', 'misspelling'} and rule['id'] in hard_rule_ids:
                issue_type = 'grammar'
            if issue_type not in HARD_TYPES:
                continue
            offset, length = match['offset'], match['length']
            start = len(utf16[:offset * 2].decode('utf-16-le'))
            end = len(utf16[:(offset + length) * 2].decode('utf-16-le'))
            out.append(dict(rule_id=rule['id'], issue_type=issue_type, start=start, end=end,
                            message=match['message']))
        return out

    def _hard_matches(self, data, text):
        return self.hard_matches(data, text, getattr(self, 'hard_rule_ids', ()))

    def screen(self, original, corrupted, edits, named_spans=()):
        lexical_checks = []
        for e in edits:
            if not e.get('requires_lexical_screen'):
                continue
            checks = {}
            for dialect in getattr(self, 'dialects', ('en-US', 'en-GB')):
                flags = []
                for word in (e['original'], e['replacement']):
                    # Lowercase carrier prevents sentence-initial name exemptions.
                    prefix = getattr(self, 'lexical_carrier', 'The word is ')
                    text = prefix + word.lower() + '.'
                    matches = self._hard_matches(self.check(text, dialect), text)
                    flags.append(any(m['issue_type'] == 'misspelling' and m['start'] == len(prefix)
                                     and m['end'] == len(prefix) + len(word) for m in matches))
                checks[dialect] = dict(source_recognized=not flags[0], replacement_nonword=flags[1])
            if not all(c['source_recognized'] and c['replacement_nonword'] for c in checks.values()):
                return None, 'checker_lexical_guard'
            lexical_checks.append(dict(rule_id=e['rule_id'],
                                       corrupted_start=e['corrupted_start'],
                                       corrupted_end=e['corrupted_end'], dialects=checks))
        language = None
        for dialect in getattr(self, 'dialects', ('en-US', 'en-GB')):
            matches = self._hard_matches(self.check(original, dialect), original)
            # Unknown proper names should not force editorial rewriting. Only
            # dictionary spelling matches wholly inside recognized names are ignored.
            matches = [m for m in matches if not (m['issue_type'] == 'misspelling'
                       and any(a <= m['start'] and m['end'] <= b for a, b in named_spans))]
            if not matches:
                language = dialect
                break
        if language is None:
            return None, 'checker_source_error'
        bad = self._hard_matches(self.check(corrupted, language), corrupted)
        at_edits = []
        for e in edits:
            hits = [m for m in bad if (
                m['issue_type'] == 'misspelling' and m['start'] == e['corrupted_start'] and m['end'] == e['corrupted_end']
                if e['corruption_type'] == 'spelling' else
                m['issue_type'] in {'grammar', 'typographical'} and m['start'] < e['corrupted_end'] and e['corrupted_start'] < m['end'])]
            if not hits:
                return None, 'checker_missed_edit'
            at_edits.append(dict(rule_id=e['rule_id'], diagnostics=hits))
        result = dict(language=language, source_hard_diagnostics=0,
                      named_entity_spelling_exemption=True, edits=at_edits)
        if lexical_checks:
            result['lexical_checks'] = lexical_checks
        return result, None

    def close(self):
        for session in self.sessions:
            session.close()
        self.db.commit()
        self.db.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup', action='store_true', required=True)
    p.add_argument('--tools-dir', default='la_output/tools')
    a = p.parse_args()
    print(json.dumps(prepare_runtime(a.tools_dir), indent=2))
