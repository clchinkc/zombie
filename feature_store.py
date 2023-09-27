

import sqlite3
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Utilities
def preprocess_text(text, stop_words):
    """Preprocess a text by lowercasing, removing punctuations and stopwords."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]


def initialize_feedback_table(conn):
    """Initialize user feedback table if it doesn't exist."""
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            document_index INTEGER,
            user_query TEXT,
            relevance_score REAL
        )
    ''')
    conn.commit()


def retrieve_feedback_data(conn):
    """Retrieve user feedback data from the SQLite database."""
    try:
        return pd.read_sql("SELECT * FROM user_feedback", conn)
    except Exception as e:
        print(f"Error retrieving feedback data: {e}")
        return pd.DataFrame(columns=["document_index", "user_query", "relevance_score"])


def train_model_based_on_feedback(conn, model, tfidf_df):
    """Refine model based on user feedback."""
    feedback_data = retrieve_feedback_data(conn)
    
    if not feedback_data.empty:
        feedback_data = feedback_data.groupby("document_index").agg({
            "relevance_score": "mean"
        }).reset_index()
        
        training_data = tfidf_df.iloc[feedback_data["document_index"]].reset_index(drop=True)
        targets = feedback_data["relevance_score"]
        weighted_data = training_data.multiply(targets, axis=0)
        model.fit(weighted_data.values)
    
    return model


# Load data and preprocess
stop_words = set(stopwords.words('english'))
document_data = pd.read_csv("feature_store_data.csv")
document_data["content"] = document_data["content"].apply(lambda x: preprocess_text(x, stop_words))

# Feature Engineering using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(document_data["content"].apply(' '.join))

# Store the TF-IDF matrix in SQLite
conn = sqlite3.connect("feature_store.db")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_sql("document_features", conn, if_exists="replace", index=False)
initialize_feedback_table(conn)

# Initial Model Training
model = NearestNeighbors(n_neighbors=5, metric="cosine")
model.fit(tfidf_df.values)

def adjust_query_vector(query_vector, feedback_data, tfidf_df):
    """Adjust the query vector based on user feedback."""
    
    # If feedback data is empty, return the original query_vector
    if feedback_data.empty:
        return query_vector

    alpha = 0.5  # Scaling factor for positive feedback
    beta = 0.25  # Scaling factor for negative feedback

    # Retrieve vectors of the docs mentioned in feedback
    feedback_vectors = tfidf_df.iloc[feedback_data['document_index'].values].values

    # Calculate positive and negative adjustments
    positive_adjustments = np.where(feedback_data['relevance_score'].values[:, None] >= 0.5,
                                    feedback_vectors, 0)
    negative_adjustments = np.where(feedback_data['relevance_score'].values[:, None] < 0.5,
                                    feedback_vectors, 0)

    # Aggregate the adjustments
    positive_feedback_vector = alpha * positive_adjustments.sum(axis=0)
    negative_feedback_vector = beta * negative_adjustments.sum(axis=0)

    # Apply adjustments
    adjusted_query_vector = query_vector + positive_feedback_vector - negative_feedback_vector

    # Normalize the adjusted query vector to have unit length
    adjusted_query_vector /= np.linalg.norm(adjusted_query_vector)

    return adjusted_query_vector


# Continuous Feedback Loop
for iteration in range(5):  # Example of 5 iterations
    print(f"\nFeedback Loop Iteration {iteration + 1}")

    model = train_model_based_on_feedback(conn, model, tfidf_df)
    
    user_query = "How does photosynthesis work?"
    preprocessed_query = ' '.join(preprocess_text(user_query, stop_words))
    query_vector = vectorizer.transform([preprocessed_query]).toarray()[0]

    # Adjust the query vector based on past feedback
    feedback_data = retrieve_feedback_data(conn)
    adjusted_query_vector = adjust_query_vector(query_vector, feedback_data, tfidf_df)

    distances, indices = model.kneighbors([adjusted_query_vector])

    print(f"Top 5 documents for the query '{user_query}':")
    for dist, index in zip(distances[0], indices[0]):
        truncated_content = document_data.iloc[index]["content"][:20]
        print(f"Relevance Score: {(1 - dist):.2f} | Content Snippet: {' '.join(truncated_content)}...")

    feedback = []
    for dist, index in zip(distances[0], indices[0]):
        truncated_content = document_data.iloc[index]["content"][:20]
        relevance_score = float(input(f"Rate the relevance of document '{' '.join(truncated_content)}...' (0 to 1): "))
        feedback.append(relevance_score)
    
    feedback_df = pd.DataFrame({
        "document_index": indices[0],
        "user_query": [user_query] * 5,
        "relevance_score": feedback
    })
    feedback_df.to_sql("user_feedback", conn, if_exists="append", index=False)

print("Feedback loop completed.")




# Depending on the scale of your application, SQLite might not be the best choice for feature storage. Distributed databases or data warehouses might be more suitable for larger-scale applications.
# A more sophisticated text search system might utilize deep learning embeddings, update features regularly based on the entire corpus, or consider other contextual information.

"""

Components Of Feature Store In Machine Learning
There are several components in a feature store- Data Transformation, Data Storage, Data Serving, ML Monitoring, and ML Feature Registry. Let us look at all these components of a feature store in detail-

Data Transformation
Think of data transformation as the magical process where raw data turns into valuable features for machine learning models. It's like a chef turning ingredients into a delicious dish! In this component, data scientists apply techniques like feature engineering, data cleaning, and preprocessing to transform raw data into meaningful features. They perform (clever tricks like) scaling, encoding, or creating new features through mathematical operations. It's where the real magic happens!

Data Storage
Data storage is like a wardrobe that keeps your favorite outfits neatly organized. In a feature store, it's where all the valuable features are stored and managed. Whether it's in-memory databases, file systems, or cloud-based solutions, the data storage component ensures that features are easily accessible, scalable, and ready to be used by machine learning models. It's like having a treasure chest of valuable features at your fingertips!

Data Serving
Data serving is the process of delivering features to the required machine learning models. This component ensures that models have quick access to the right features during training or inference- it's all about efficiency and speed! Data serving ensures that models receive the required features promptly, allowing them to make accurate predictions or decisions.

Model Monitoring
ML monitoring is like having a watchful eye over your machine learning models- tracking model performance, their training data sources, data drift, and potential issues. Just as you monitor your pet's health or your favorite sports team's performance, ML model monitoring ensures that models perform at their best. It alerts you if something goes wrong, like a sudden drop in accuracy or unexpected changes in your data sources or distribution. Think of it as having a personal assistant that keeps your models in check!

Feature Registry
The ML feature registry (like a librarian) meticulously catalogs and keeps track of all the commonly used features and feature definitions by machine learning models. It's like having a detailed index of all the important books in a library. The ML feature registry helps you maintain version control, track changes, and organize your features. It ensures you can easily refer to previous versions, compare their impact on model performance, and maintain consistency. It acts as your trusted guide through the feature factory!

Think of all these components like a well-coordinated team-

data transformation creates the features,

data storage stores them,

data serving delivers them,

ML monitoring keeps an eye on them, and

the ML feature registry keeps everything organized.

Together, these components form the backbone of a robust and efficient feature store, enabling data experts to build successful machine-learning models!


"""


"""
A feature store is a centralized repository for storing, sharing, and accessing machine learning features. It ensures that the features used during training are the same as those used in serving or prediction. Although feature stores are primarily designed for machine learning use-cases, they can be leveraged in a text search program to maintain consistent feature extraction, storage, and retrieval.

Here's a step-by-step guide on how to use a feature store in a text search program:

1. **Feature Extraction**:
    - Depending on your search program, extract relevant features from the text. This could be anything from tokenized words, named entity recognition (NER) tags, sentiment scores, TF-IDF values, embeddings, etc.
    
2. **Storing Features in the Feature Store**:
    - Instead of directly indexing the text (as in traditional search engines like Elasticsearch), index the extracted features in the feature store. This ensures a consistent set of features across training and serving.

3. **Text Search**:
    - When a query comes in, extract the same set of features from the query text as you did for your documents.
    - Query the feature store for relevant matches based on the extracted query features. This could involve:
        - Looking up feature values or embeddings that closely match the query features.
        - Using pre-trained models to score and rank documents based on these features.
    - Return the matched documents or information to the user.

4. **Continuous Learning**:
    - As users interact with your search system, collect feedback on which results are relevant and which aren't.
    - Use this feedback as labeled data to train (or fine-tune) a model to predict relevance based on the features in the feature store.
    - Update the model periodically to reflect new data and insights.

5. **Serving with Consistency**:
    - Whenever your search program is updated, or a new model is trained, you can ensure that the features it uses for making predictions are the same as those used during training by sourcing them from the feature store.
    - This ensures consistency and reproducibility in your search program.

6. **Monitoring and Maintenance**:
    - Periodically monitor the features for drift or changes. If the way features are extracted or the nature of the data changes, it may be necessary to update the feature definitions in the feature store.
    - Use tools provided by the feature store for monitoring and managing the lifecycle of features.

Some popular feature store options include AWS SageMaker Feature Store, Tecton, Feast, among others. Each of these solutions comes with its own set of APIs and utilities, so the implementation details might vary. However, the general concept outlined above will remain the same.

By integrating a feature store into your text search program, you can ensure consistency in feature extraction and usage across different stages of your application, potentially leading to more accurate and reproducible search results.
"""


"""
Open-Source Feature Store Tools And Frameworks
Several open-source feature stores are available in the ML industry, each bringing its unique value to the table, catering to different needs and use cases. Understanding their strengths and use cases helps data scientists and ML engineers select the most suitable tool for their projects. Let us explore some open-source feature store tools/frameworks used by data scientists and ML engineers for managing features in ML projects.

1. Hopsworks
The Hopsworks feature store stands out with its comprehensive features, making it an all-in-one solution. It offers a user-friendly interface that simplifies feature engineering, model management, serving, and collaboration. It also supports popular ML frameworks like TensorFlow and PyTorch. It is applicable in a wide range of use cases, from fraud detection and recommendation systems to image recognition and natural language processing. Companies like Intel and Siemens use the Hopsworks feature store to streamline their ML workflows and boost their business operations.

2. Feast
Introduced by Gojek in collaboration with Google Cloud, Feast excels in online feature store serving and management. It ensures the smooth delivery of features to ML models by offering scalable and efficient feature serving. Feast provides advanced features like feature validation, data quality monitoring, and online/offline feature store synchronization. It is suitable for real-time serving in applications such as personalized recommendations, dynamic pricing, and anomaly detection. Companies like Netflix and Twilio use the Feast feature store to streamline their ML workflows and boost business operations.

3. Tecton
Tecton, introduced by Tecton.ai, is another popular feature store focusing on high-performance feature engineering and serving. It offers a declarative approach to feature engineering, allowing users to define features using a high-level language. This simplifies feature generation and reduces time spent on feature engineering tasks. Tecton also provides data lineage tracking, ensuring transparency and even data governance. It is suitable for use cases that require complex feature engineering, such as customer churn prediction, fraud detection, and demand forecasting. Companies like Spotify and Salesforce use the Tecton feature store to streamline their ML workflows and boost business operations.
"""


"""
Integrated Feature Store Machine Learning
Integrated Machine Learning Feature Stores, such as Azure Feature Store and AWS Sagemaker Feature Store, offer seamless and scalable solutions for managing and serving ML features within ML workflows. By centralizing feature management, these feature stores allow for easy development and deployment of ML models, enabling data science and ML experts to focus on extracting insights from data and delivering accurate predictions.

Let us look at a few popular integrated feature stores you can use for your ML projects-

Azure ML Feature Store
Microsoft introduced Azure Feature Store as a fully managed feature storage and management service in Azure Machine Learning. It aims to simplify and streamline the feature engineering process by providing a centralized and scalable repository for storing, discovering, and sharing features across multiple teams and projects. Azure Feature Store enables data scientists and ML engineers to manage and utilize features efficiently, enabling faster model development, improved collaboration, and enhanced model accuracy within the Azure ecosystem.

Key Features of Azure ML Feature Store
Below are some key features of the Azure ML Feature Store-

Integration With Azure Machine Learning- Azure Feature Store seamlessly integrates with Azure Machine Learning, allowing data scientists and ML engineers to easily access and consume features during model training and inference workflows. It offers a unified experience within the Azure ecosystem.

Feature Discovery And Exploration- This feature store offers capabilities for discovering and exploring features, allowing users to search for specific features, explore feature metadata, and understand the distribution and statistics of features. This helps in feature selection and analysis.

Real-time And Batch Serving- Azure Feature Store supports real-time and batch serving of features, allowing models to access features in real-time for online inference or retrieve historical feature data for offline training. This flexibility caters to various use cases and applications.

Azure ML Feature Store Use Cases
Let us look at a few suitable use cases for the Azure Feature Store-

Personalized Recommendations- Azure Feature Store can store and serve customer-related features, enabling the development of customized recommendation systems that deliver relevant suggestions to users.

Fraud Detection- By storing and serving ML features related to user behavior, transaction history, and suspicious patterns, Azure Feature Store helps build fraud detection models to quickly identify fraudulent activities.

AWS Sagemaker Feature Store
The AWS Sagemaker Feature Store was introduced by Amazon Web Services (AWS) as a part of its Sagemaker machine learning platform. It aims to simplify and streamline the management and sharing of features for ML projects, offering a centralized and scalable feature store solution. The AWS Sagemaker Feature Store offers a managed and scalable feature storage and access solution that helps enhance the productivity and collaboration of data scientists and ML engineers. With its serverless architecture, data versioning capabilities, and real-time and batch serving support, it is ideally used for a wide range of ML use cases within the AWS ecosystem.

Key Features of Azure ML Feature Store
Below are some key features of the Azure ML Feature Store-

Fully Managed And Serverless- AWS Sagemaker Feature Store is a fully managed service that eliminates manual provisioning and scaling. It operates serverless, allowing users to focus on feature engineering and model development without worrying about the underlying infrastructure.

Data Versioning And Lineage- AWS Sagemaker Feature Store offers built-in data versioning and lineage tracking capabilities. It allows users to track the history of feature data, understand applied transformations, and ensure ML experiments' reproducibility.

Real-time And Batch Serving- The feature store supports both real-time and batch serving of features, allowing models to access online features in real-time for online inference or retrieve historical feature data (offline features) for offline training. This flexibility caters to various use cases, whether real-time recommendation systems or batch-based model retraining.

Azure ML Feature Store Use Cases
Let us look at a few suitable use cases for the Azure Feature Store-

Demand Forecasting- AWS Sagemaker Feature Store can be leveraged to store and serve features related to historical sales data, pricing, inventory, and external factors. This enables accurate demand forecasting models that help optimize inventory management and supply chain planning.

Risk Assessment And Fraud Detection- By storing and serving features related to customer behavior, transaction history, and fraud indicators, the feature store facilitates the development of risk assessment and fraud detection models to identify and prevent fraudulent activities in real-time.

GCP Feature Store
Google Cloud introduced the GCP Feature Store in June 2021 as a purpose-built, managed service for feature storage and management within the Google Cloud Platform (GCP) ecosystem. It offers a centralized solution to simplify the process of feature engineering, storage, data governance, and access for ML projects. The GCP Feature Store enables a unified and efficient ML workflow with its seamless integration with other GCP services such as BigQuery, Dataflow, and AI Platform. The GCP Feature Store has exceptional capabilities like data versioning, lineage tracking, real-time and batch serving capabilities, and feature monitoring, enabling users to effectively manage and use features in their machine-learning models.

Key Features of GCP Feature Store
Below are some key features of the GCP Feature Store-

Native Integration With GCP- The GCP Feature Store seamlessly integrates with other GCP services, such as BigQuery, Dataflow, and AI Platform, enabling an efficient and streamlined ML workflow. It leverages the power of Google's infrastructure to deliver robust and reliable feature (online) storage and retrieval.

Data Versioning And Lineage- GCP Feature Store's built-in versioning and lineage tracking capabilities allow users to trace the history of features, understand transformations, and ensure reproducibility and compliance. It makes auditing and debugging of ML pipelines easier.

Real-time And Batch Serving- The feature store supports real-time and batch serving features, making it suitable for many use cases. It enables real-time inference by serving the latest feature data and batch processing for training models with historical feature values.

GCP Feature Store Use Cases
Let us look at a few suitable use cases for the Azure Feature Store-

Recommendation Systems- You can use the GCP Feature Store for storing and serving user-related features, item metadata, and user interactions. This helps develop personalized recommendation systems that deliver relevant suggestions in real-time, boosting user experience and engagement.

Health Monitoring And Analysis- Healthcare providers can use the GCP Feature Store to store and serve patient health-related features, including vital signs, medical history, etc. By building ML models that leverage these features, healthcare professionals can monitor patient health in real-time, identify potential health risks, and take prompt actions.

Databricks Feature Store
The Databricks Feature Store was introduced as part of the Databricks Unified Analytics Platform in 2021, offering a seamless and integrated feature storage and management solution. Databricks Feature Store is built on top of Apache Spark and Delta Lake, and many companies, including Uber, Spotify, and Airbnb, use it. It is available as part of the Databricks Runtime for Machine Learning, a fully managed, cloud-based platform for ML. It is fully integrated with other components of Databricks, such as the Databricks MLflow and the Databricks Workspace Model Registry.

Key Features of Databricks Feature Store
Below are some unique features of the Databricks Feature Store you should know-

Point-in-time Lookups- The Databricks Feature Store supports point-in-time lookups, meaning you can retrieve the feature values for a particular time. This can be useful for training models on historical data or for understanding how a feature has changed over time.

Databricks MLflow Integration- The Databricks Feature Store is integrated with Databricks MLflow, a platform for managing the ML lifecycle. This integration makes it easy to track the lineage of features and deploy models to production. Other feature stores do not typically have this level of integration with MLflow.

Time-Series Features- The Databricks Feature Store supports time series features, which means you can store features associated with a timestamp. This support makes it easy to create and manage time series features and to perform time series analysis for ML tasks that require time series data, such as forecasting and anomaly detection. Other feature stores do not typically have this level of support for time series features.

Online Feature Stores- The Databricks Feature Store supports online feature stores, meaning features can be served in real-time. This is helpful for ML applications that require real-time predictions, such as fraud detection and recommendation engines.

Databricks Feature Store Use Cases
Here are some interesting real-world use cases for the Databricks Feature Store-

Anomaly Detection- You can use the Databricks Feature Store to build anomaly detection models by leveraging its centralized repository for storing and managing normal and abnormal behavior features. This will help you quickly and easily develop and deploy anomaly detection models and ensure that the same features are used for training and inference.

Customer Segmentation- You can use the Databricks Feature Store to segment customers by providing a centralized repository for storing and managing features related to customer behavior. This helps improve the efficiency of marketing campaigns by ensuring that the right messages are sent to the right customers.
"""


"""
Feature Stores In ML- Project Ideas
Here are a few interesting project ideas to help you understand the implementation of feature stores in ML-

1. FEAST Feature Store Example For Scaling Machine Learning
This project will show you how to use FEAST Feature Store to manage, store, and discover features for customer churn prediction problems. You will work with a customer churn dataset with eight features- Created_at, Customer_id, Churned, Category, Sex, Age, Order gmv, Credit type. Working on this project will teach you about the Feast Architecture and Entities and Feature view in Feast. You will also learn about online stores, offline stores, data retrieval, and several commands in Feast. You will learn how to create training data using Feast. In this project, you will work with Random Forest and the Gradient Boosting models and then generate real-time predictions using online features and/or offline features from Feast by deploying it via Postman.

Source Code- https://www.projectpro.io/project-use-case/feast-feature-store-example-for-scaling-machine-learning

2. E-Commerce Recommendation Engine Using AWS Sagemaker Feature Store
In this project, you will build a real-time recommendation engine, combining a Collaborative Filtering model and an XGBoost model, for an e-commerce website using a synthetic online grocer dataset. You will use Amazon SageMaker Feature Store (both online and offline) to store feature data for model training, validation, and real-time inference. The recommendation engine will recommend top products that a customer will likely purchase while browsing through the e-commerce website based on the customer's purchase history, real-time click stream data, and other customer profile data. 

Source Code- https://github.com/aws-samples/sagemaker-feature-store-real-time-recommendations

3. ML Research Paper Recommendation System
For this project, you will develop an NLP-based recommender system to help users discover valuable machine-learning papers that align with their interests. For this project, you will use the Hopsworks feature store and Trubrics' feedback component with Streamlit. You will scrape new arXiv papers data from the arXiv website and store them in the Hopsworks features store. You will also generate sentence embeddings from the selected features and then generate recommendations using the Streamlit app.

Source Code- https://github.com/yassine-rd/PaperWhiz/tree/master
"""
