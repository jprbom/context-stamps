"""Use only observed query facets; document stamps remain exactly 32 bytes."""

from context_stamps import FacetQuery, Family, HashingEncoder, SphericalStamp, Stamp256Codec

encoder = HashingEncoder(64)
families = {name: Family(encoder.identity, 64, 64, 17 + i)
            for i, name in enumerate(('content', 'entity', 'intent', 'task'))}
codec = Stamp256Codec(families)
document = codec.encode({name: encoder.encode(text) for name, text in {
    'content': 'Update timeout', 'entity': 'worker_alpha', 'intent': 'implement', 'task': 'timeout',
}.items()})

# The caller only knows the content query. Do not invent entity/intent/task values.
query = FacetQuery(SphericalStamp.encode({'content': encoder.encode('Update timeout')},
                                        {'content': families['content']}))
print('Observed facets:', query.compare(document))
print('Ranking utility, not confidence:', query.score(document))
print('Calibration scope:', query.policy_scope('demo:' + codec.schema.identity))
assert len(codec.pack(document)) == 32
assert set(query.compare(document)) == {'content'}
# Keep exact authorization and any required entity/task checks outside this score.
