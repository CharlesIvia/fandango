# Question to answer - Is there a conflict of interest for a website that sells both movie tickets and displays movie reviews?

# Does Fandango display artificially higher than average reviews?

# Fandango has two ratings:

# STATS - rating in stars 0-5 displayed on their website HTML
# RATING - Actual true rating numerically shown on movie's page

# Required imports
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
fan_df = pd.read_csv("fandango_scrape.csv")

print(fan_df.head())
print(fan_df.info())
print(fan_df.describe())

# Explore the relationship between popularity of a filtm and its rating

plt.figure(figsize=(10, 5))
sns.scatterplot(data=fan_df, x="RATING", y="VOTES")
plt.show()

# Calculate the correlation between the columns/attributes

corr_mat = fan_df.corr()
print(corr_mat)

# Create a new column that is able to strip the year from the title strings and set this new column as YEAR

fan_df["YEAR"] = fan_df["FILM"].str[-6:].str[1:-1]
print(fan_df.head())

# How many movies are in the Fandango DataFrame per year?

num_movies_per_year = fan_df["YEAR"].value_counts()
print(num_movies_per_year)

# Visualize the count of movies per year with a plot

sns.countplot(data=fan_df, x="YEAR")
plt.show()

# What are the 10 movies with the highest number of votes?

votes_filter = fan_df["VOTES"].sort_values(ascending=False).head(10).index
highest_votes = fan_df.loc[votes_filter]
print(highest_votes)

# How many movies have zero votes?

zero_votes = fan_df["VOTES"] == 0
print(zero_votes.sum())

# Create DataFrame of only reviewed films by removing any films that have zero votes.

reviewed_df = fan_df[fan_df["VOTES"] > 0]
print(reviewed_df.info())

# Create a KDE plot (or multiple kdeplots) that displays the distribution of
# ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5
plt.figure(figsize=(10, 5), dpi=100)
sns.kdeplot(data=reviewed_df, x="RATING", fill=True, clip=[0, 5], label="True Rating")
sns.kdeplot(
    data=reviewed_df, x="STARS", fill=True, clip=[0, 5], label="Stars Displayed"
)
plt.legend(loc=(0.05, 0.85))
plt.show()

# Difference between True rating and stars displayed

reviewed_df["STARS_DIFF"] = round(reviewed_df["STARS"] - reviewed_df["RATING"], 1)

print(reviewed_df.head())

# Create a count plot to display the number of times a certain difference occurs
plt.figure(figsize=(11, 5), dpi=100)
sns.countplot(data=reviewed_df, x="STARS_DIFF", palette="magma")
plt.show()

# We can see from the plot that one movie was displaying over a 1 star
# difference than its true rating! What movie had this close to 1 star differential?
one_star_diff = reviewed_df["STARS_DIFF"] == 1
print(reviewed_df[one_star_diff])

## Comparison of Fandango ratings to other sites

all_sites = pd.read_csv("all_sites_scores.csv")
print(all_sites.head())
print(all_sites.info())
print(all_sites.describe())

# Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews.
plt.figure(figsize=(11, 5), dpi=100)
sns.scatterplot(data=all_sites, x="RottenTomatoes", y="RottenTomatoes_User")
plt.show()

# Create a new column based off the difference between critics ratings and users ratings for Rotten Tomatoes.


all_sites["RottenTomatoes_Diff"] = (
    all_sites["RottenTomatoes"] - all_sites["RottenTomatoes_User"]
)

print(all_sites.head())

# Calculate the Mean Absolute Difference between RT scores and RT User scores

mean_abs_diff = all_sites["RottenTomatoes_Diff"].apply(abs).mean()
print(mean_abs_diff)

# Plot the distribution of the differences between RT Critics Score and RT User Score.
plt.figure(figsize=(12, 6), dpi=100)
sns.histplot(data=all_sites, x="RottenTomatoes_Diff", kde=True)
plt.title("RT Critics Score minus RT User Score")
plt.show()

# Create a distribution showing the absolute value difference between Critics and Users on Rotten Tomatoes.
all_sites["RottenTomatoes_Diff_Abs"] = all_sites["RottenTomatoes_Diff"].apply(
    lambda x: abs(x)
)
plt.figure(figsize=(12, 6), dpi=100)
sns.histplot(data=all_sites, x="RottenTomatoes_Diff_Abs", kde=True)
plt.title("Abs Difference btwn RT Critics Score and RT User Score")
plt.show()

# What are the top 5 movies users rated higher than critics on average?

higher_than_critic = all_sites.nsmallest(5, "RottenTomatoes_Diff")[
    ["FILM", "RottenTomatoes_Diff"]
]
print(higher_than_critic)

# Show the top 5 movies critics scores higher than users on average.

lower_than_critic = all_sites.nlargest(5, "RottenTomatoes_Diff")[
    ["FILM", "RottenTomatoes_Diff"]
]
print(lower_than_critic)

## Meta critic

# Display a scatterplot of the Metacritic Rating versus the Metacritic User rating.

plt.figure(figsize=(11, 5), dpi=100)
sns.scatterplot(data=all_sites, x="Metacritic", y="Metacritic_User")
plt.show()

## IMDB

# Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB.

plt.figure(figsize=(11, 5), dpi=100)
sns.scatterplot(
    data=all_sites, x="Metacritic_user_vote_count", y="IMDB_user_vote_count"
)
plt.show()

# What movie has the highest IMDB user vote count

highest_votes_IMDB = all_sites.nlargest(1, "IMDB_user_vote_count")
print(highest_votes_IMDB)

# What movie has the highest Metacritic User Vote count?

highest_votes_Metacritic = all_sites.nlargest(1, "Metacritic_user_vote_count")
print(highest_votes_Metacritic)

##Fandango Scores vs. All Sites

df = pd.merge(fan_df, all_sites, how="inner", on="FILM")
print(df.info())
print(df.head())

# Normalize columns to Fandango STARS and RATINGS 0-5

df["RT_Norm"] = np.round(df["RottenTomatoes"] / 20, 1)
df["RTU_Norm"] = np.round(df["RottenTomatoes_User"] / 20, 1)

df["Meta_Norm"] = np.round(df["Metacritic"] / 20, 1)
df["Meta_U_Norm"] = np.round(df["Metacritic_User"] / 2, 1)

df["IMDB_Norm"] = np.round(df["IMDB"] / 2, 1)

print(df.head())

# create a norm_scores DataFrame that only contains the normalizes ratings.

norm_scores = df[
    ["STARS", "RATING", "RT_Norm", "RTU_Norm", "Meta_Norm", "Meta_U_Norm", "IMDB_Norm"]
]

print(norm_scores.head())

# Create a plot comparing the distributions of normalized ratings across all sites
# Function to move legend


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
sns.kdeplot(data=norm_scores, clip=[0, 5], shade=True, palette="Set1", ax=ax)
move_legend(ax, "upper left")
plt.show()

# Create a KDE plot that compare the distribution of RT critic ratings against the STARS displayed by Fandango.
fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
sns.kdeplot(
    data=norm_scores[["RT_Norm", "STARS"]],
    clip=[0, 5],
    shade=True,
    palette="Set1",
    ax=ax,
)
move_legend(ax, "upper left")
plt.show()

# Create a histplot comparing all normalized scores.
fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
sns.histplot(norm_scores, bins=50)
plt.show()

# Based off the Rotten Tomatoes Critic Ratings, what are the top 10 lowest rated movies?

norm_films = df[
    [
        "STARS",
        "RATING",
        "RT_Norm",
        "RTU_Norm",
        "Meta_Norm",
        "Meta_U_Norm",
        "IMDB_Norm",
        "FILM",
    ]
]

bad_films = norm_films.nsmallest(10, "RT_Norm")
print(bad_films)

# Visualize the distribution of ratings across all sites for the top 10 worst movies.
fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
worst_films = norm_films.nsmallest(10, "RT_Norm").drop("FILM", axis=1)
sns.kdeplot(data=worst_films, clip=[0, 5], shade=True, palette="Set1")
plt.title("Ratings for RT Critic's 10 Worst Reviewed Films")

plt.show()
