[Azure Data Explorer](https://docs.microsoft.com/en-us/azure/data-explorer/kusto/query/) (Kusto) is a fully-managed big data analytics cloud platform and data-exploration service, developed by Microsoft, that ingests structured, semi-structured (like JSON) and unstructured data (like free-text). As part of this platform, Microsoft introduced a SQL-like language named Kusto Query Language (KQL), which enables querying and visualizing of the data. Unlike SQL, KQL can not be used for inserting, updating or deleting data. KQL is mainly used for querying over Azure's analytics and monitoring systems, Log Analytics and Application Insights which both use ADX as their big-data analytics service.

KQL queries vary in their complexity which can effect the time it takes for them to run, ranging from a few milliseconds to more than five minutes. Some queries are so complex that they are often canceled by the ADX server after ten minutes of wait time. Identifying query complexity without running the query can benefit both ADX server side and users using KQL. ADX clusters are mostly shared by multiple users creating a problem of noisy neighbours where users using many complex queries can impact the ADX for other users which use light queries. Being able to predict the complexity of a query can be used to throttle over usage by users. Additionally, predicting complexity can help save users a vast amount of time by preventing them from running queries that will timeout and help them refine too complex queries.
In this work we present a model which aims to evaluate query runtime duration and complexity based on the query and the data container context. For that purpose we introduce, to our knowledge, the first dataset of KQL queries along with their runtime duration and container context collected from Log Analytics and Application Insights backend services. 

<a id="contents"></a>
# Contents
<!-- MarkdownTOC -->

- [Contents](#contents)
- [Setup](#setup)
- [Usage](#usage)

<!-- /MarkdownTOC -->

<a id="setup"></a>
# Setup
- Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed.
- In repo's source directory run:
  <pre>conda env create
  conda activate kql_nlp</pre>

<a id="usage"></a>
# Usage

<pre>
DATA_PATH=./output 

python main.py \
--output_path=$OUTPUT_PATH \
--batch_size=32 \
--epochs=1 \
--with_cuda=True \
--cuda_device='0' \
--lr=2e-5 \
--download_data=False \
--seed=42 \
--buckets_strategy="uniform" \
--buckets_count=5 \
--minimal_queries_per_container=100 \
--data_parallel=True \
--outliers_threshold=0.995 \
--use_weights=False \
--inference=False \
--dropout=0.1 \
--dl_worker_count=4
</pre>