import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import missingno as msno
from matplotlib.ticker import MaxNLocator

filepath='C:/Users/Shabana/OneDrive/Desktop/test/sentimentdataset.csv'
df = pd.read_csv(filepath)

#Lets remove unnecessary columns
df.drop(df.columns[[0,1]], axis=1, inplace=True)
df.drop(columns=['User', 'Text'], axis=1, inplace=False)

#Printing necessary details
print(df.head())
print(df.dtypes)
print(df.shape)

#Lets make sure there are no missing values
msno.matrix(df.sample(250))
plt.show()

#Let's focus on the latest trends
df = df.loc[df['Year'] == 2023]

#Let modify the hashtags column to keep the relevant part only
regex = r'^[^#]*(#[^#]*)#'
df['Hashtags'] = df['Hashtags'].str.extract(regex)[0]

#Lets remove leading and trailing spaces from country, sentiment, hashtags, platform column values for effective evaluation
df['Country'] = df['Country'].str.strip()
df['Sentiment'] = df['Sentiment'].str.strip()
df['Hashtags'] = df['Hashtags'].str.strip()
df['Platform'] = df['Platform'].str.strip()

#Lets find the most retweeted and liked sentiments and hashtags per country and platform
##Sum of retweets and likes based on sentiment, hashtags and country as a group
grouped = df.groupby(['Sentiment', 'Platform', 'Hashtags', 'Country'])
total_retweets = grouped['Retweets'].sum().sort_values(ascending=False)
total_likes = grouped['Likes'].sum().sort_values(ascending=False)

# Define the mean value for each country and platform
country_retweet_mean = total_retweets.groupby(['Country', 'Platform']).mean().to_dict()
country_like_mean = total_likes.groupby(['Country', 'Platform']).mean().to_dict()

#Filter out tweets and likes based on the mean values of tweets and likes for a country as well as platform
def filter_by_mean(row, mean_function):
    country_platform = (row['Country'], row['Platform'])
    # Check if the (Country, Platform) combination exists in the dictionary
    if mean_function == "mean_like":
        if country_platform in country_like_mean:
            return row['Likes'] > country_like_mean[country_platform]
    elif mean_function == "mean_retweet":
        if country_platform in country_retweet_mean:
            return row['Retweets'] > country_retweet_mean[country_platform]
    return False

total_retweets_df = total_retweets.reset_index()
popular_tweets = total_retweets_df[total_retweets_df.apply(lambda row: filter_by_mean(row, "mean_retweet"), axis=1)]
total_likes_df = total_likes.reset_index()
most_liked = total_likes_df[total_likes_df.apply(lambda row: filter_by_mean(row, "mean_like"), axis=1)]

#Find the most retweeted and liked sentiments and hashtags per country and platform
most_retweets_platform_countrywise = popular_tweets.sort_values('Retweets', ascending=False).groupby(['Country', 'Platform']).head(5).sort_values('Country', ascending=True).reset_index(drop=True)
most_liked_platform_countrywise = most_liked.sort_values('Likes', ascending=False).groupby(['Country', 'Platform']).head(5).sort_values('Country', ascending=True).reset_index(drop=True)
with pd.ExcelWriter(r'C:\Users\Shabana\OneDrive\Desktop\test\PopularSentimentsforCountry.xlsx', engine='openpyxl') as writer:
    most_retweets_platform_countrywise.to_excel(writer, sheet_name='Most Retweets', index=False)
    most_liked_platform_countrywise.to_excel(writer, sheet_name='Most Liked', index=False)

#lets visualize the country wise popular sentiments through bar plot

##Convert series to DataFrames
most_retweets_platform_countrywise_df = most_retweets_platform_countrywise.reset_index(drop=True)
most_liked_platform_countrywise_df = most_liked_platform_countrywise.reset_index(drop=True)

##Merging both DataFrames on the 'Sentiment', 'Hashtags', 'Country', 'Platform' columns
merged_df = pd.merge(most_retweets_platform_countrywise_df, most_liked_platform_countrywise_df, on=['Sentiment', 'Hashtags', 'Country', 'Platform'], how='outer')

##Melt the DataFrame to reshape it for use with FacetGrid
melted_data = merged_df.melt(id_vars=['Sentiment', 'Country', 'Platform'], value_vars=['Likes', 'Retweets'],
                      var_name='Metric', value_name='Count')
##Define the order for Sentiment and Metric
sentiment_order = merged_df['Sentiment'].unique()
metric_order = ['Likes', 'Retweets']

##Set up the FacetGrid
g = sns.FacetGrid(melted_data, col="Platform", row="Country", margin_titles=True)

##Map the barplot onto the grid
g.map(sns.barplot,  'Sentiment', 'Count', 'Metric', dodge=True, palette='Set1', order=sentiment_order, hue_order=metric_order)

##Add titles and adjust labels
g.set_axis_labels("Sentiment", "Count")
g.set_titles(col_template="{col_name} Platform", row_template="{row_name} Country")

g.fig.tight_layout()

g.add_legend(title="Metric", loc='upper right')

##Rotate x-axis labels for better readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)  # Rotate labels by 90 degrees for readability
    ax.figure.tight_layout()  # Automatically adjust subplots to fit labels

##Loop through each axis to add counts on top of bars
for ax in g.axes.flat:
    for p in ax.patches:  # Iterate through the bars
        # Get the height of each bar
        height = p.get_height()
        # Add the count value as text on top of the bar
        ax.text(p.get_x() + p.get_width() / 2, height, f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)

# Adjust the spacing between subplots and margins
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.2)  # Adjust the bottom to give space for labels

# Show the plot
plt.show()

#Count plot for top 5 sentiments for each platform across all the countries
grouped = df.groupby(['Sentiment', 'Hashtags', 'Platform'])
total_retweets = grouped['Retweets'].sum().sort_values(ascending=False)
total_likes = grouped['Likes'].sum().sort_values(ascending=False)

## Define the mean value for each platform
platform_retweet_mean = total_retweets.groupby('Platform').mean()
platform_like_mean = total_likes.groupby('Platform').mean()

##Convert series to DataFrames
total_retweets_df = total_retweets.reset_index()
total_likes_df = total_likes.reset_index()

##select top 5 sentiments for a platform
top_5_retweets_df = (total_retweets_df[total_retweets_df['Retweets'] > total_retweets_df['Platform'].map(platform_retweet_mean)]).sort_values('Retweets', ascending=False).groupby('Platform').head(5)
top_5_likes_df = (total_likes_df[total_likes_df['Likes'] > total_likes_df['Platform'].map(platform_like_mean)]).sort_values('Likes', ascending=False).groupby('Platform').head(5)
with pd.ExcelWriter(r'C:\Users\Shabana\OneDrive\Desktop\test\TopFiveforPlatform.xlsx', engine='openpyxl') as writer:
    top_5_retweets_df.to_excel(writer, sheet_name='Top 5 Retweets', index=False)
    top_5_likes_df.to_excel(writer, sheet_name='Top 5 Liked', index=False)

##Merging both DataFrames on the 'Sentiment', 'Hashtags' and 'Platform' columns
merged_df = pd.merge(top_5_retweets_df, top_5_likes_df, on=['Sentiment', 'Hashtags', 'Platform'], how='outer')

## Melt the DataFrame to have 'Likes' and 'Retweets' in the same column
df_melted = pd.melt(merged_df, id_vars=['Sentiment', 'Platform'],
                    value_vars=['Likes', 'Retweets'],
                    var_name='Metric', value_name='Count')

##Define the order for Sentiment and Metric
sentiment_order = merged_df['Sentiment'].unique()
metric_order = ['Likes', 'Retweets']

##Create a FacetGrid with subplots for each platform
g = sns.FacetGrid(df_melted, col="Platform", margin_titles=True)

##Map the bar plot with 'Sentiment' on x-axis, 'Count' on y-axis, and 'Metric' as hue
g.map(sns.barplot, 'Sentiment', 'Count', 'Metric', dodge=True, palette='Set1',
      order=sentiment_order, hue_order=metric_order)

##Rotate x-axis labels for better readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)  # Rotate labels by 90 degrees for readability

##Adjust the titles and labels
g.set_axis_labels('Sentiment', 'Count')
g.set_titles(col_template="{col_name} Platform")

## Adjust the spacing between subplots and margins
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.2)

plt.show()

# Lets find the correlation between the numeric columns

##Convert the numeric columns to number type to calculate correlation coefficient
df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).apply(pd.to_numeric, errors='coerce')

##Correlation between numeric columns
numeric_features = df.select_dtypes(include=[np.number])
coff = numeric_features.corr()[['Retweets', 'Likes']]
print(coff)