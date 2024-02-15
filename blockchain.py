import hashlib
import time


class Vote:
    def __init__(self, candidate):
        self.candidate = candidate

class Voter:
    def __init__(self, voter_id_hash):
        self.voter_id_hash = voter_id_hash

class Block:
    def __init__(self, index, previous_hash, timestamp, votes, voter_ids, hash_):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.votes = votes
        self.voter_ids = voter_ids
        self.hash = hash_

    def compute_hash(self):
        block_data = (str(self.index) + str(self.previous_hash) +
                      str(self.timestamp) + str([vote.__dict__ for vote in self.votes]) +
                      str([voter.__dict__ for voter in self.voter_ids]))
        return hashlib.sha256(block_data.encode('utf-8')).hexdigest()

    @staticmethod
    def create_genesis_block():
        return Block(0, '0', time.time(), [], [], Block.compute_hash(Block(0, '0', time.time(), [], [], '0')))

def simple_encrypt_vote(vote_data, key):
    # In a real-world scenario, stronger encryption methods like AES would be preferable
    return ''.join([chr(ord(vote_data[i]) + ord(key[i % len(key)])) for i in range(len(vote_data))])

def simple_decrypt_vote(encrypted_vote, key):
    return ''.join([chr(ord(encrypted_vote[i]) - ord(key[i % len(key)])) for i in range(len(encrypted_vote))])

def validate_blockchain(blockchain):
    for i in range(1, len(blockchain)):
        current_block = blockchain[i]
        previous_block = blockchain[i-1]

        if current_block.previous_hash != previous_block.hash:
            return False
        if current_block.hash != current_block.compute_hash():
            return False
    return True

def add_vote_to_blockchain(candidate, voter_id_hash, blockchain, encryption_key):
    encrypted_vote = simple_encrypt_vote(candidate, encryption_key)
    encrypted_voter_id = simple_encrypt_vote(voter_id_hash, encryption_key)
    new_vote = Vote(encrypted_vote)
    new_voter_id = Voter(encrypted_voter_id)
    last_block = blockchain[-1]
    new_block = Block(last_block.index + 1, last_block.hash, time.time(), [new_vote], [new_voter_id], '0')
    new_block.hash = new_block.compute_hash()

    if validate_blockchain(blockchain):
        blockchain.append(new_block)
        print(f"Block {new_block.index} added to the blockchain.")
    else:
        print("Failed to add block: Blockchain validation failed.")

def cast_vote(voter_id, candidate, blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes):
    current_time = time.time()
    if current_time > voting_deadline:
        print("Voting failed: Voting deadline has passed.")
        return
    if candidate not in allowed_candidates:
        print(f"Voting failed: '{candidate}' is not a valid candidate.")
        return
    voter_id_hash = hashlib.sha256(voter_id.encode('utf-8')).hexdigest()
    if voter_id_hash not in eligible_voter_hashes:
        print("Voting failed: Voter is not eligible.")
        return
    if any(voter.voter_id_hash == simple_encrypt_vote(voter_id_hash, encryption_key) for block in blockchain for voter in block.voter_ids):
        print("Voting failed: Duplicate vote detected.")
        return
    add_vote_to_blockchain(candidate, voter_id_hash, blockchain, encryption_key)
    print(f"Vote for '{candidate}' cast successfully.")

def tally_votes(blockchain, decryption_key):
    vote_count = {}
    for block in blockchain:
        for vote in block.votes:
            decrypted_vote = simple_decrypt_vote(vote.candidate, decryption_key)
            if decrypted_vote in vote_count:
                vote_count[decrypted_vote] += 1
            else:
                vote_count[decrypted_vote] = 1
    print("Final Vote Tally:")
    for candidate, count in vote_count.items():
        print(f"{candidate}: {count}")
    return vote_count

blockchain = [Block.create_genesis_block()]
encryption_key = "secret_key"
decryption_key = encryption_key
allowed_candidates = ["CANDIDATE_A", "CANDIDATE_B"]
eligible_voter_hashes = [
    hashlib.sha256('VOTER_ID_1'.encode('utf-8')).hexdigest(),
    hashlib.sha256('VOTER_ID_2'.encode('utf-8')).hexdigest(),
]
voting_deadline = time.time() + 10  # Extend the deadline for testing

# 1. Successful Vote Casting
print("Testing successful vote casting...")
cast_vote('VOTER_ID_1', 'CANDIDATE_A', blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes)

# 2. Duplicate Voting Attempt
print("Testing duplicate voting attempt...")
cast_vote('VOTER_ID_1', 'CANDIDATE_B', blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes)

# 3. Invalid Candidate Voting Attempt
print("Testing invalid candidate voting attempt...")
cast_vote('VOTER_ID_2', 'INVALID_CANDIDATE', blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes)

# 4. Ineligible Voter Voting Attempt
print("Testing ineligible voter voting attempt...")
cast_vote('VOTER_ID_3', 'CANDIDATE_A', blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes)

# 5. Late Voting Attempt
print("Testing late voting attempt...")
time.sleep(11)  # Delay to simulate voting after the deadline
cast_vote('VOTER_ID_3', 'CANDIDATE_A', blockchain, encryption_key, allowed_candidates, voting_deadline, eligible_voter_hashes)

# Wait a moment before tallying to ensure the late vote has been processed
time.sleep(1)

# 6. Blockchain Integrity Check Post-Voting
print("Checking blockchain integrity post-voting...")
if validate_blockchain(blockchain):
    print("Blockchain integrity check passed.")
else:
    print("Blockchain integrity check failed.")

# 7. Vote Tallying Accuracy
print("Tallying votes for accuracy check...")
tally_votes(blockchain, decryption_key)

# Print the blockchain data for inspection
for block in blockchain:
    print("Block:", block.index, "Hash:", block.hash, "Previous Hash:", block.previous_hash, "Votes:", [vote.__dict__ for vote in block.votes])

# PKI for Vote Encryption and Decryption:
# Each voter is assigned a public/private key pair.
# Votes are encrypted with the voter's public key before being added to a block.
# A centralized or decentralized authority holds the private key to decrypt votes only after the voting period ends.

# Merkle Trees for Vote Verification:
# Instead of storing individual votes directly in the block, store a Merkle root of all votes.
# Each vote is a leaf in the Merkle Tree, allowing efficient and secure verification of vote integrity without revealing vote details.

# Proof of Authority (PoA) for Consensus:
# Validators are selected based on a reputation system within the voting context).
# Selected validators are responsible for adding new blocks to the blockchain, ensuring a distributed and secure consensus process.

# Multiple votes in one block