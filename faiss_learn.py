import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


import faiss
d=64
index = faiss.IndexFlatIP(d)  
print(index.is_trained)
index.add(xb)              
print(index.ntotal) 


k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


import faiss

# Load your document dataset
documents = ["This is the first document.", "This is the second document."]

# Create a Faiss index
index = faiss.IndexFlatL2(128)

# Add your documents to the index
for document in documents:
    index.add(document)

# Save the index
index.save("index.bin")