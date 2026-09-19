"""Offline exact-first routing and reusable, revocable context handoffs."""

from context_stamps.routing import ProgressiveRouter
from context_stamps.session import ContextSession
from context_stamps.workflow import ContextNode

session = ContextSession()
session.put(ContextNode('implementation', 'TIMEOUT = 10', 'v1', frozenset({'developer'})))
session.put(ContextNode('requirement', 'Timeout must be ten seconds.', 'v1', frozenset({'developer'})))
session.link('implementation', 'requirement', 'depends_on', provenance='review-1')
versions = {'implementation': 'v1', 'requirement': 'v1'}

def precise(ids, limit):
    # Replace with an authorized dense/hybrid backend for semantic queries.
    return [('implementation', 1.0)][:limit]

result = ProgressiveRouter().search(eligible=['implementation'], exact_key='implementation',
                                    precise=precise, scope='demo:v1', limit=1)
packet, receipt, reused = session.issue(result['ids'], role='developer', revisions=versions)
assert not reused and len(receipt) == 32
again, same_receipt, reused = session.issue(result['ids'], role='developer', revisions=versions)
assert reused and packet == again and receipt == same_receipt
assert session.resolve(receipt, role='developer', revisions=versions) == packet
print('Route:', result['route'], 'Evidence bytes:', packet.units, 'Receipt bytes:', len(receipt))
print('Repeated evidence construction reused:', reused)

session.put(ContextNode('requirement', 'Timeout must be twenty seconds.', 'v2', frozenset({'developer'})))
assert session.resolve(receipt, role='developer', revisions=versions).status == 'insufficient'
print('Source update invalidated the previous receipt.')
