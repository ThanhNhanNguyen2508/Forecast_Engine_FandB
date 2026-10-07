"""One activation per declared shipment, with stable exact display allocation."""
from collections import defaultdict

def shipment_group(offer):
    return offer.source_terms.get('shipment_group_id', offer.offer_id)

def allocate_delivery_fees(lines, offers):
    groups=defaultdict(list)
    for line in lines: groups[shipment_group(offers[line.offer_id])].append(line)
    allocations={}
    for key,rows in groups.items():
        fees={offers[l.offer_id].delivery_cost for l in rows}
        if len(fees)!=1: raise ValueError('INCONSISTENT_SHIPMENT_FEE')
        fee=fees.pop(); rows=sorted(rows,key=lambda l:l.offer_id)
        share=round(fee/len(rows),2)
        for i,l in enumerate(rows): allocations[l.offer_id]=share if i<len(rows)-1 else fee-share*(len(rows)-1)
    return allocations
