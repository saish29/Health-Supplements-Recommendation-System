import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Preprocessing

df = pd.read_csv("Health_Supplements.csv")

# Dropping NA Values
df = df.dropna().reset_index(drop = True)

# Dropping unnamed:0 columns 

df.drop(["Unnamed: 0"], axis = 1, inplace = True)

# Drop duplicate products from product column

df = df.drop_duplicates(subset = ['Product'], keep = 'first').reset_index(drop = True)

# Feature Engineering

feature_vector = ["Type of supplement", "Consumer", "Flavour", "Category", "Amount", "Vegetarian"]

df['combined_features'] = df[feature_vector].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF Vectorizer

tfid_vec = TfidfVectorizer()

tfid_matrix = tfid_vec.fit_transform(df['combined_features'])

# Similarly Calcualtion

cosine_sim = linear_kernel(tfid_matrix, tfid_matrix)

# Recommendation Engine

def get_recommendations(product_index, consumer_filter=None):
    sim_scores = list(enumerate(cosine_sim[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get the top 5 similar products (excluding the product itself)
    product_indices = [i[0] for i in sim_scores]

    if consumer_filter:
        filtered_indices = [idx for idx in product_indices if df.iloc[idx]['Consumer'] == consumer_filter]
        return df.iloc[filtered_indices]
    else:
        return df.iloc[product_indices]



# Step 6: Deploy the Streamlit Interface
def main():
    st.title('Health Supplement Recommendation Engine')

    # Get user input
    consumer_type = st.selectbox('Select Consumer Type', ['All', 'Men', 'Women', 'Kids'])
    supplement_type = st.text_input('Enter Type of Supplement')
    category = st.text_input('Enter Category')
    flavour = st.text_input('Enter Flavour')

    # Filter the data based on user input
    filtered_data = df.copy()

    if consumer_type != 'All':
        filtered_data = filtered_data[filtered_data['Consumer'] == consumer_type]

    if supplement_type:
        filtered_data = filtered_data[filtered_data['Type of supplement'].str.contains(supplement_type, case=False)]

    if category:
        filtered_data = filtered_data[filtered_data['Category'].str.contains(category, case=False)]

    if flavour:
        filtered_data = filtered_data[filtered_data['Flavour'].str.contains(flavour, case=False)]

    if len(filtered_data) == 0:
        st.warning('No matching products found.')

    # Show recommendations
    if len(filtered_data) > 0:
        st.subheader('Recommended Products')
        for idx, row in filtered_data.iterrows():
            st.write(f"Product: {row['Product']}, Rating: {row['Rating']}, Price: {row['Price']}")

if __name__ == '__main__':
    main()