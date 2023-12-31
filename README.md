# Steel Price Forecasting

## Data

Data is price values of steel in INR (Indian rupees) on each given day from 2015-01-01 to 2023-12-26 gotten from [Source of data](https://www.investing.com/commodities/ncdex-steel-futures-historical-data). And can be downloaded through the source or from this repository from a file **"Steel Futures Historical Data"**.

Data has 7 variables:

1. Date 
2. Price
3. Open (Opening price)
4. High (Highest price during that day)
5. Low (Lowest price during that day)
6. Vol. (Volume sold)
7. Change % (Price change)

_**For this project the focus will be on the Price variable given Date.**_

**Example of data:**
|Date      |Price     |Open      |High      |Low       |Vol.      |Change %  |
|----------|----------|----------|----------|----------|----------|----------|
|2023-01-03|39000     |41000     |41000     |38000     |0.43K     | -0.23%   |
|2023-01-02|41000     |40000     |42500     |39700     |0.09K     | +0.25%   |
|2023-01-01|40000     |39000     |41000     |38500     |0.23K     | +0.13%   |

**Graph of Price to Index** \
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/priceToIndex.png" width="500" height="400">

**Graph of Price to Date** \
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/priceToDate.png" width="500" height="400">

## Preprocessing

Data is pretty good from the start but some steps which needs to be done to use the data for forecasting.

1. Remove unnecessary variables
2. Set Date as index
3. Change values of Price, Open, ... from String to Float
4. Handle missing values

**STEP 1:**\
Variable Vol. was the only variable that had nan values in it. Also it was not necessary for the project thus it was removed to simplify data

**STEP 2:**\
Date was set as index to simplify code. This was done by formatting string "26/12/2023" -> "2023-12-26" using pandas.to_datetime() function.

**STEP 3:**\
Numerical values on variables [Price, Open, High, Low] were in a string of "41,000.00" which need to be formatted to 41000.00. This was done by removing the unnecessary "," and then transforming the value from String to Int.

**STEP 4.:**\
Data has missing values between 2017-2021 like seen in **Graph of Price to Date** so data was changed to start from the first value on 2021. New graph can be seen below.

**Data from 2021 onwards**\
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/priceToDateUpdated.png" width="500" height="400">

## Predictions

I began by fitting different polynomials to the **whole** data using the index to see what polynomial would fit the best and what parameter values it would give. Then I moved into using ARMA to find the best parameters for this current situtation.

### Fitting a line

Five lines were fitted to the data by using polynomials and linear regression. Polynomials follow a equation of ax+b, ax^2+bx+c and so on.

**Fitted lines on the data**\
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/LinesOnPriceToIndex.png" width="500" height="400">

From the graph it can be seen that polynomials with degrees of 3 and 4 are the best. Since they are pretty close to each other it's best to choose 3 degree polynomial since it is simpler.\ This polynomial follow equation of **ax^3+bx^2+c+d**, where\
**a = 0.000101\
b = -0.212\
c = 96.258\
d = 38903.300**

### ARMA
Data was split into training and test sets like shown in the picture below. This way we can train the model and test it to find the best values for ARMA by calculating the RSME.

**Data split into training and test sets**\
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/TrainAndTest.png" width="500" height="400">

First ARMA model was done with parameters of (1,0,1) that granted a RMSE of 1532.58, how this fits in to the graph is shown below.
**Default parameters (1,0,1) as prediction**\
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/predictionDefault.png" width="500" height="400">

Then parameters were tested from [0,5] for each parameter and the lowest RMSE of 795.06 was found for parameters of (1,3,3) and how this fits into the data is shown below.
**Optimized parameters (1,3,3) as prediction**\
<img src="https://github.com/DaniBarlund/SteelPriceForecasting/blob/main/photos/predictionOptimized.png" width="500" height="400">

## Conclusion

Project was a good introduction to ARMA as a method of forecasting but in the future projects I will be taking a closer look into other methods such as SARIMA which should take into account the seasonal changes in data.
