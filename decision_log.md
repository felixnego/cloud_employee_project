# Decision Log and Assumptions 

- **Assumption:** audience biometric data has to deal with facial expressions based on hints in the assignment document.
- Audience data will be synthesized.
- We will use DuckDB as the main repository layer for this exercise with the important note that this won't do in a production setup, with possibly more (concurrent) writers. However, parquet on disk works for the PoC due to the columnar and ready-heavy nature of the video processing pipeline.
- We are going to use `Docker` (with `docker-compose` if needed) to keep the dependency management clean. This is especially needed for the video processing pipeline, since it has multiple Python dependencies.
- **Assumption:** we will not handle, in this PoC, the case of ingesting new (video) data properly. However, we will build necessary integrations to satisfy the secondary, technical user.
- **Assumption**: given that the main user can use SQL or Python to self-serve their insights, decided that the main UI layer should be a Jupyter Notebook. In a production environment, it is possible that can still be the case, with a separate visualisation layer (dashboards) added on top of the data warehouse.

---
- Video features were selected based on a quick research of tools that can comfortably run on my local machine. In a production pipeline, the features would be carefully selected and defined by Subjetc Matter Experts (SMEs).
- Important Note: Alongside SMEs, a production version of the infrastructure would have defined and implemented quality gates and tests for the video features extracted. 
- Important Note: In this PoC the pipeline goes through the `extractors` and gets all the video features from the `.mp4` files. This is **not** the architecture I would recommend for production, which would most likely entail a job queue and parallel processing (either my model/feature or by file).
---
- There is a separate DuckDB schema for the generated audience data. In this way, real data can be easily plucked it by any engineer wishing to develop on top of this. 
- In order to suppor the above, there is a contract in `audience/contract.py` where I define the tables for the audience data. 
- Audience data is not randomly generated. This adds complexity to the repo for the sake of making the notebook more realistic. The audience data is generated from the audience creative features.
---
- The assignment also mentions the following:

```
We then pair it with user generated demographics and survey responses. Add in a psychological test and viola! Those are (most of) our raw ingredients.
```

- In this PoC this type of data was not mocked / generated, due to time constraints and already existing complexity.
---
- For the secondary user, the `understand` part is covered in the `README`, the schema and `docs` and the architectural diagram.
- An engineer can integrate with this work by versioning the notebook and developing on top of it directly. 
- Another way the secondary user can integrate or build upon this PoC is via the 2 APIs we expose: one for new data ingestion and one for using the video processing pipeline separately, async in any other context (e.g.: as a step in a wider pipeline or a job queue, etc). **Important Note:** the APIs are not production-ready and missing many features necessary for real use. 
- For simplicity, creatives arrive by URL, not multipart upload to signed S3 bucket (recommended). 