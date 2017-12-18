# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:58:39 2017

@author: Georgie
"""

"""
We're going to learn about a product recommender engine from yhat's blog
http://blog.yhat.com/posts/trending-products-recommender-engine.html.
Each row in the data represents the number of cart adds for a particular
product on a particular day.
"""

#Import necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator

# Load the data into a data frame
df = pd.read_csv('sample-cart-add-data.csv')

# Use apivot table to group by ID & Age
cart_adds = pd.pivot_table(df, values='count', index=['id', 'age'])

# Plot a particular example
# Note the age is in negative numbers (-5 days old) so it reads from left
# to right
ID = 542
trend = np.array(cart_adds.loc[ID])
plt.figure(1)
x = np.arange(-len(trend),0)
plt.plot(x, trend, label="Cart Adds 542")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(ID))
plt.show()


ID = 92
trend = np.array(cart_adds.loc[ID])
plt.figure(2)
x = np.arange(-len(trend),0)
plt.plot(x, trend, label="Cart Adds 92")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(ID))
plt.show()




#The following is a smoothing function
def smooth(series, window_size, window):
    # Generate data points 'outside' of x on either side to ensure
    # the smoothing window can operate everywhere
    ext = np.r_[2 * series[0] - series[window_size-1::-1],
                series,
                2 * series[-1] - series[-1:-window_size:-1]]
    weights = window(window_size)
    weights[0:int(window_size/2)] = np.zeros(int(window_size/2))
    sm = np.convolve(weights / weights.sum(), np.reshape(ext, len(ext)), mode='same')
    return sm[window_size:-window_size+1]  # trim away the excess data

smoothed = smooth(trend,7,np.hamming)

#plt.plot(x, smoothed, label="Smoothed")


#Standardise the data by dividing by the IQR
def standardize(series):
    iqr = np.percentile(series, 75) - np.percentile(series, 25)
    return (series - np.median(series)) / iqr

# We can now compare products
ID = [542, 92]
plt.figure(3)

for id in ID:
    trend = np.array(cart_adds.loc[id])
    x = np.arange(-len(trend),0)    
    trend_std = standardize(trend)
    plt.plot(x, trend_std,label="Cart Adds"+str(id))
    smoothed_std = standardize(smooth(trend,7,np.hamming))
    plt.plot(x, smoothed_std, label="Smoothed"+str(id))

plt.show()   
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    

#Calculating the slope
ID = 542
trend = np.array(cart_adds.loc[ID])
x = np.arange(-len(trend)+1,0)
smoothed_std = standardize(smooth(trend,7,np.hamming))
plt.figure(4)
slopes = smoothed_std[1:]-smoothed_std[:-1]
plt.plot(x, slopes)




#Combining everything to find the top trending products, by analysing the slope:

#Choose the type of smoothing
SMOOTHING_WINDOW_FUNCTION = np.hamming
SMOOTHING_WINDOW_SIZE = 7


def train():
    df = pd.read_csv('sample-cart-add-data.csv')
    df.sort_values(by=['id', 'age'], inplace=True)
    trends = pd.pivot_table(df, values='count', index=['id', 'age'])

    trend_snap = {}

    for i in np.unique(df['id']):
        trend = np.array(trends.loc[i])
        smoothed = smooth(trend, SMOOTHING_WINDOW_SIZE, SMOOTHING_WINDOW_FUNCTION)
        nsmoothed = standardize(smoothed)
        slopes = nsmoothed[1:] - nsmoothed[:-1]
        # I blend in the previous slope as well, to stabalize things a bit and
        # give a boost to things that have been trending for more than 1 day
        if len(slopes) > 1:
            trend_snap[i] = slopes[-1] + slopes[-2] * 0.5
    return sorted(trend_snap.items(), key=operator.itemgetter(1), reverse=True)




trending = train()
print("Top 5 trending products:")
for i, s in trending[:5]:
    print("Product %s (score: %2.2f)" % (i, s))