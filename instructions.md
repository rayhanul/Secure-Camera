
## Overall architecture :


Camera -> YOLO -> TSN/UDP -> C2 -> TransReID -> Weaviate -> MongoDB




## Important Code modification 

### Edge node 

Both edge nodes in Sai’s original implementation use Redis Queue (RedisQ) for communication. In our setup, we have completely removed RedisQ. Instead, we use a TSN-based network manager to send frames directly to the receiver. The implementation details of this communication can be found in `networkManager.py`.

From Sai’s work, we only reuse the frame processing component. This has been integrated into our existing implementation.

To ensure the YOLO model works correctly, make sure that the `yolov8n.pt` file is available in the appropriate directory.

### Server node 


The server unit (C2) is responsible for receiving frames from the edge node over the TSN network and performing further processing. Unlike Sai’s original implementation, which relies on Redis Queue (RedisQ) for data transfer, our system directly receives frames using a TSN-based network manager.

Upon receiving the data, the server processes the frames using the integrated frame processor and extracts features using the TransReID model. The extracted embeddings are then stored and managed using Weaviate for similarity search and MongoDB for persistent storage.




The communication logic for receiving data is handled within the network management component, while the processing pipeline combines both the reused components from Sai’s work and the newly integrated modules.

However, there is another important change in the model. while the Sai's work use `vit_transreid_market.pth` as model, we use `transformer_best.pth` as i did not find that particular model in the internet. 



### Set database 

To make the database setup, we generate two docker-compose file avialable in the directory db_mongo and db_vector. 


 Run command 


```
sudo docker compose up -d
```

to set weaveate vector database. you will see something like 

```
0.0.0.0:8080->8080/tcp
0.0.0.0:50051->50051/tcp

```


It also use Mongo Db to store . 


Similary go the directory db_mongo inside C2 and run the same command as above. You should see MongoDB running on port 27017




### Set environment variables: 

Run the following command to setup required env variables. 

```
export MONGO_HOST_URL="localhost:27017"
export MONGO_ROOT_USERNAME="admin"
export MONGO_ROOT_PASSWORD="secret123"
export WEAVIATE_URL="http://localhost:8080"
export MONGO_DB_NAME="reid_db"
export MONGO_COLLECTION_NAME="reid_collection"

```

You can update `~/.bashrc` file on ubuntu if you like to make these changes permanent. 





## Important code fix 

1. Ensure the env variables are set prior to running exp 
2. Place the models and weights in the right directory
3. Use TSN network manager instead of Redis Queue 
4. 


### Running example 
To run the edge node :

use the following command, if you like to modify the params:


```
python main.py \
  --camera_index 0 \
  --model_path /path/to/yolov8n.pt \
  --dest_ip 192.168.10.11 \
  --dest_port 12345 \
  --priority 7
```
However, you can also the following simple command if you wish to run with default setting:

```
python main.py 

```
To run the server node, run the following command:

```
python main.py \
  --listen_ip 0.0.0.0 \
  --port 12345 \
  --weaviate_url http://localhost:8080 \
  --transreid_model_path /path/to/TransReID/weights/vit_transreid_market.pth

  ```


Similarly, you can run only `python main.py` as well. 



