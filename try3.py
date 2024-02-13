
import hashlib
import time


class Vote:
    def __init__(self, voter_id, candidate):
        self.voter_id = voter_id
        self.candidate = candidate

class Block:
    def __init__(self, index, previous_hash, timestamp, votes, hash_):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.votes = votes
        self.hash = hash_

    def compute_hash(self):
        block_data = (str(self.index) + str(self.previous_hash) + 
                      str(self.timestamp) + str([vote.__dict__ for vote in self.votes]))
        return hashlib.sha256(block_data.encode('utf-8')).hexdigest()

    @staticmethod
    def create_genesis_block():
        return Block(0, '0', time.time(), [], Block.compute_hash(Block(0, '0', time.time(), [], '0')))

def add_vote_to_blockchain(vote, blockchain):
    last_block = blockchain[-1]
    new_block = Block(last_block.index + 1, last_block.hash, time.time(), [vote], '0')
    new_block.hash = new_block.compute_hash()
    blockchain.append(new_block)

blockchain = [Block.create_genesis_block()]

def cast_vote(voter_id, candidate, blockchain):
    vote = Vote(voter_id, candidate)
    add_vote_to_blockchain(vote, blockchain)

# Example usage:
cast_vote('VOTER_ID_1', 'CANDIDATE_A', blockchain)
cast_vote('VOTER_ID_2', 'CANDIDATE_B', blockchain)
cast_vote('VOTER_ID_3', 'CANDIDATE_A', blockchain)

# Print the blockchain data
for block in blockchain:
    print("Block:", block.index, "Votes:", [vote.__dict__ for vote in block.votes])
