import os

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Replace with your actual connection string.
# For local: 'mongodb://localhost:27017/'
# For Atlas: 'mongodb+srv://<user>:<password>@cluster.mongodb.net/'
def list_mongo_data():
    mongo_host_url = os.environ["MONGO_HOST_URL"]
    mongo_user = os.environ["MONGO_ROOT_USERNAME"]
    mongo_pass = os.environ["MONGO_ROOT_PASSWORD"]
    CONNECTION_STRING = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host_url}"

    try:
        # 1. Connect to the MongoDB Client
        client = MongoClient(CONNECTION_STRING)

        # Check if connection is successful
        # The ismaster command is cheap and does not require auth.
        client.admin.command("ismaster")
        print("Connected successfully to MongoDB server!")

        # 2. Select the Database and Collection
        db = client[os.environ["MONGO_DB_NAME"]]
        collection = db[os.environ["MONGO_COLLECTION_NAME"]]

        # 3. Retrieve Data
        # .find() returns a cursor (an iterator)
        cursor = collection.find()

        # 4. Iterate and Print Data
        print(f"\nListing documents in '{collection}':")
        print("-" * 30)

        count = 0
        for document in cursor:
            count += 1

        if count == 0:
            print("No documents found in this collection.")
        else:
            print(f"found {count} documents")

    except ConnectionFailure:
        print("Server not available. Please check your connection string.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 5. Close the connection
        if "client" in locals():
            client.close()

            client.close()


if __name__ == "__main__":
    list_mongo_data()
