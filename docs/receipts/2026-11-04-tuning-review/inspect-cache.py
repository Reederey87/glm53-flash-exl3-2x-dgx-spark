from pathlib import Path
import hashlib,json,datetime
root=Path('/root/.cache/b12x/compile/preparation')
rows=[]
for p in sorted(root.glob('*.json')):
    raw=p.read_bytes();d=json.loads(raw);identity=d['identity'];records=d['records']
    assert identity['schema_version']==22
    assert identity['compute_capability']==[12,1] and identity['sm_count']==48
    assert identity['measurement']=='stream_gated_events_v1'
    coverage=[]
    for k,r in records.items():
        c=r['coverage'];assert set(c)=={'cartesian_count','legal_count','effective_count','measured_count'}
        assert all(type(v) is int and v>=0 for v in c.values())
        assert c['cartesian_count']>=c['legal_count']>=c['effective_count']==c['measured_count']>1
        coverage.append(c)
    rows.append({'filename':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':p.stat().st_mtime_ns,'schema':identity['schema_version'],'tuning_cache_version':identity['tuning_cache_version'],'compute_capability':identity['compute_capability'],'sm_count':identity['sm_count'],'measurement':identity['measurement'],'namespace_sha256':hashlib.sha256(json.dumps(identity['namespace'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'record_count':len(records),'coverage_totals':{k:sum(c[k] for c in coverage) for k in ('cartesian_count','legal_count','effective_count','measured_count')},'all_cached_races_complete':True})
print(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':str(root),'files':rows,'limitation':'Cached completed races only; does not enumerate fixed, heuristic, missing, or unused model plans. No GPU work or cache writes.'},indent=2))
