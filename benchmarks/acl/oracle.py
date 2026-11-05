#!/usr/bin/env python3
"""Deterministic ACL corpus and reference decisions, policy A in README.md."""
import argparse
import hashlib
import json
from pathlib import Path

SEED = 0x5EED42

def corpus():
    users = [dict(id=0, role='admin', tenant=0, team=None)]
    users += [dict(id=i, role='doctor', tenant=0, team=(i - 1) % 10 + 1) for i in range(1, 41)]
    users += [dict(id=i, role='nurse', tenant=0, team=1) for i in range(41, 50)]
    state = SEED
    visits = []
    for i in range(10000):
        values = []
        for _ in range(4):
            state ^= (state << 13) & 0xffffffff
            state ^= state >> 17
            state ^= (state << 5) & 0xffffffff
            state &= 0xffffffff
            values.append(state)
        visits.append(dict(id=i, tenant=int(values[0] % 10 == 0), owner=11 + values[1] % 30,
                           team=1 + values[2] % 10, archived=values[3] % 10 == 0))
    return users, visits

def decision(u, v):
    if u['role'] == 'admin':
        return u['tenant'] == v['tenant']
    if u['role'] != 'doctor' or v['archived']:
        return False
    if u['id'] == v['owner']:
        return True
    return u['team'] is not None and u['team'] == v['team']

def verify_cases():
    v = dict(tenant=1, owner=7, team=3, archived=False)
    u = dict(id=7, role='doctor', tenant=0, team=None)
    assert decision(u, v)  # owner works without a team, including cross-tenant
    assert not decision({**u, 'id': 8}, v)
    assert decision({**u, 'id': 8, 'team': 3}, v)
    assert not decision({**u, 'role': 'nurse', 'team': 3}, v)
    assert not decision(u, {**v, 'archived': True})
    assert decision({**u, 'role': 'admin', 'tenant': 1}, {**v, 'archived': True})
    assert not decision({**u, 'role': 'admin'}, v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/acl'))
    args = parser.parse_args()
    verify_cases()
    users, visits = corpus()
    bits = bytearray((len(users) * len(visits) + 7) // 8)
    allows = 0
    for ui, user in enumerate(users):
        for vi, visit in enumerate(visits):
            if decision(user, visit):
                n = ui * len(visits) + vi
                bits[n // 8] |= 1 << (n % 8)
                allows += 1
    data = json.dumps({'seed': hex(SEED), 'users': users, 'visits': visits}, separators=(',', ':'), sort_keys=True).encode()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'dataset.json').write_bytes(data)
    (args.output / 'oracle.bin').write_bytes(bits)
    print(json.dumps({'users': len(users), 'visits': len(visits), 'decisions': len(users)*len(visits), 'allow': allows,
                      'dataset_sha256': hashlib.sha256(data).hexdigest(), 'oracle_sha256': hashlib.sha256(bits).hexdigest(),
                      'policy': 'specification JavaScript example; see benchmarks/acl/README.md'}, indent=2))

if __name__ == '__main__':
    main()
