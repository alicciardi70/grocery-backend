def build_chroma_db_if_needed():
    import os
    import pandas as pd
    import chromadb
    from chromadb.utils import embedding_functions

    persist_directory = "./grocery_chroma_db"
    input_file = "grocery_products_clean.tsv"

    if os.path.exists(persist_directory):
        print("✅ Chroma DB already exists.")
        return

    print("🛠️ Building Chroma vector DB from TSV...")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="grocery_products", embedding_function=embed_fn)

    df = pd.read_csv(input_file, delimiter='\t', encoding='utf-8')
    total_rows = len(df)
    batch_size = 200
    documents_batch = []
    metadatas_batch = []
    ids_batch = []

    for idx, row in df.iterrows():
        name = str(row['Product Name']).strip()
        price_raw = str(row['Price']).strip().replace('$', '')
        try:
            price_value = float(price_raw)
        except ValueError:
            price_value = 0.0
        store = str(row['Store']).strip() if pd.notna(row['Store']) else "Unknown Store"
        brand = str(row['Brand']).strip() if pd.notna(row['Brand']) else "Unknown Brand"

        documents_batch.append(name)
        metadatas_batch.append({"price": price_value, "store": store, "brand": brand})
        ids_batch.append(str(idx))

        if len(documents_batch) >= batch_size or idx == total_rows - 1:
            collection.add(documents=documents_batch, metadatas=metadatas_batch, ids=ids_batch)
            documents_batch, metadatas_batch, ids_batch = [], [], []

        if idx % 500 == 0 or idx == total_rows - 1:
            print(f"Progress: {(idx + 1) / total_rows * 100:.2f}%")

    print("✅ Chroma vector DB built.")
