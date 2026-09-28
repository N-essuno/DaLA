"""Resource-configured lexical backends shared across language packs."""
import importlib.util
from pathlib import Path
from threading import Lock


class SerializedDictionary:
    """Guard a native dictionary handle that cannot be shared across threads."""
    def __init__(self, lookup):
        self._lookup = lookup
        self._lock = Lock()

    def lookup(self, word):
        with self._lock:
            return self._lookup(word)



def load_dictionary(root, config):
    if config.get('backend', 'hunspell') == 'hunspell':
        from spylls.hunspell import Dictionary
        return Dictionary.from_files(str(root / 'index'))
    if config['backend'] == 'voikko':
        spec = importlib.util.spec_from_file_location('dala_voikko', root / config['python_binding'])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        import ctypes
        for library in config.get('preload_libraries', []):
            ctypes.CDLL(str(root / library), mode=ctypes.RTLD_GLOBAL)
        module.Voikko.setLibrarySearchPath(str(root / config['library_directory']))
        engine = module.Voikko(config['language'], path=str(root / config['dictionary_directory']))
        return SerializedDictionary(engine.spell)
    raise ValueError('Unsupported lexical backend')
