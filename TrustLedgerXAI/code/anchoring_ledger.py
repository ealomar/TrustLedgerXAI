"""
A software model of the Fabric chaincode and validation semantics that the
TrustLedger-XAI protocol depends on. It is NOT a Fabric deployment: it has no
consensus, no network, no signatures and no timing. It models exactly three
things, because the protocol's guarantees depend on them:

  1. endorsement: a transaction is endorsed only if at least one endorser is
     an organisation other than the submitting (generating) one, and the
     chaincode rejects, at simulation time, a key already present in committed
     state (write-once);
  2. MVCC validation: each endorsed transaction carries the version of the key
     it read during simulation; at commit, a transaction whose read version no
     longer matches committed state is marked invalid. This is what makes
     write-once hold when two first-writes to the same key are endorsed
     concurrently, before either has committed;
  3. ordering: committed entries record the block height at which they were
     committed. Heights give an order between commitments; they are not
     clock times (Fabric block headers carry no timestamp, and the transaction
     timestamp is supplied by the submitting client).

The world-state value stored per commitment is the canonical JSON of
{"H": <explanation digest>, "h_b": <record digest>}; the block height is
metadata of the committing block, not part of the value the chaincode writes.
"""

import rfc8785


class MVCCLedger:
    def __init__(self, orgs, generating_org, controlled_by_generator=()):
        self.orgs = set(orgs)
        self.gen = generating_org
        self.controlled = set(controlled_by_generator) | {generating_org}
        self.state, self.version, self.pending = {}, {}, []
        self.height, self.invalid = 0, 0
        self.validate_reads = True

    def value_bytes(self, H, h_b):
        return rfc8785.dumps({"H": H, "h_b": h_b})

    def submit(self, key, H, h_b, creator, endorsers):
        """Simulate + endorse. Raises on policy or write-once violation."""
        if not (set(endorsers) & (self.orgs - self.controlled)):
            raise PermissionError("endorsement policy not satisfied")
        if key in self.state:
            raise PermissionError("write-once: key already committed")
        read_version = self.version.get(key)          # None: key absent
        self.pending.append((key, H, h_b, read_version))

    def cut_block(self):
        self.height += 1
        for key, H, h_b, rv in self.pending:
            if self.validate_reads and self.version.get(key) != rv:
                self.invalid += 1                     # MVCC_READ_CONFLICT
                continue
            self.state[key] = {"H": H, "h_b": h_b, "height": self.height}
            self.version[key] = self.height
        self.pending = []

    def query(self, key):
        return self.state.get(key)
