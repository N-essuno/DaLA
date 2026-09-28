"""Agent-authored smoke controls, not native-validated linguistic gold."""
import sys,json,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_european_pairs import request,validate
from dfm12.jobs import query
samples=[('de','Das Kind spielt im Garten.','Das Kind spielen im Garten.'),('fr','Les enfants jouent dans le jardin.','Les enfants joue dans le jardin.'),('es','Los niños juegan en el jardín.','Los niños juega en el jardín.'),('it','I bambini giocano in giardino.','I bambini gioca in giardino.'),('cs','Děti si hrají na zahradě.','Děti si hraje na zahradě.'),('pt-PT','As crianças brincam no jardim.','As crianças brinca no jardim.'),('fi','Lapset leikkivät puistossa.','Lapset leikkii puistossa.'),('et','Lapsed mängivad aias.','Lapsed mängib aias.'),('ca','Els nens juguen al jardí.','Els nens juga al jardí.'),('el','Τα παιδιά παίζουν στον κήπο.','Τα παιδιά παίζει στον κήπο.'),('ro','Copiii se joacă în grădină.','Copiii se joacăă în grădină.'),('uk','Діти граються в саду.','Діти грається в саду.')]
work=[]
for lang,good,bad in samples:
 for label,a,b in [('injected_error',good,bad),('unchanged',good,good),('reversed',bad,good)]:
  work.append(dict(language=lang,original=a,corrupted=b,control=label))
def run(item):
 i,r=item
 return dict(**r,result=validate(query(f'http://127.0.0.1:{8600+i%8}/v1',request(r))))
with concurrent.futures.ThreadPoolExecutor(16) as pool: results=list(pool.map(run,enumerate(work)))
from pathlib import Path
p=Path('/work/mimir/DaLA/la_output/european_audit/controls_20260927.json');p.write_text(json.dumps(results,ensure_ascii=False,indent=2))
for r in results:print(r['language'],r['control'],r['result']['decision'],r['result']['reason'])
assert all(r['result']['decision']!='pass' for r in results if r['control']!='injected_error')
