# Decision Log and Assumptions 

- **Assumption:** audience biometric data has to deal with facial expressions based on hints in the assignment document.
- Audience data will be synthesized.
- We will use DuckDB as the main repository layer for this exercise with the important note that this won't do in a production setup, with possibly more (concurrent) writers. However, parquet on disk works for the PoC due to the columnar and ready-heavy nature of the video processing pipeline.
- We are going to use `Docker` (with `docker-compose` if needed) to keep the dependency management clean. This is especially needed for the video processing pipeline, since it has multiple Python dependencies.
- **Assumption:** we will not handle, in this PoC, the case of ingesting new video data. 

---
- Video features were selected based on a quick research of tools that can comfortably run on my local machine. In a production pipeline, the features would be carefully selected and defined by Subjetc Matter Experts (SMEs).
- Important Note: Alongside SMEs, a production version of the infrastructure would have defined and implemented quality gates and tests for the video features extracted. 
- Important Note: In this PoC the pipeline goes through the `extractors` and gets all the video features from the `.mp4` files. This is **not** the architecture I would recommend for production, which would most likely entail a job queue and parallel processing (either my model/feature or by file).
---
