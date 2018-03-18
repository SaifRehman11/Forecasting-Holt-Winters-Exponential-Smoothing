library(forecast)
library(smooth)
library(TStools)

#Load the Data
medium_noise <- read.csv("medium_noise.csv", header = FALSE)

#Convert to Time Series
medium_noise <- ts(medium_noise, frequency = 12, start = c(2012,1))

#Set horizon and number of rolling origins
h <- 12
origins <- 10
medium_noise_length <- length(medium_noise)

train_length <- medium_noise_length - h - origins + 1
test_length <- h + origins - 1

medium_noise_train <- ts(medium_noise[1:train_length], 
                         frequency = frequency(medium_noise),
                         start = start(medium_noise))

medium_noise_test <- medium_noise[(train_length+1):medium_noise_length]

medium_noise_forecasts <- matrix(NA, nrow = origins, ncol = h)
medium_noise_holdout <- matrix(NA, nrow = origins, ncol = h)

colnames(medium_noise_forecasts) <- paste0("horizon",c(1:h))
rownames(medium_noise_forecasts) <- paste0("origin", c(1:origins))

dimnames(medium_noise_holdout) <- dimnames(medium_noise_forecasts)

View(medium_noise_holdout)


for(i in 1:origins) {
  #Create a ts object out of the medium noise data
  our_train_set <- ts(medium_noise[1:(train_length+i-1)],
                      frequency = frequency(medium_noise),
                      start = start(medium_noise))
  
  #Write down the holdout values from the test set
  medium_noise_holdout[i,] <- medium_noise_test[i-1+(1:h)]
  
  #Produce forecasts and write them down
  medium_noise_forecasts[i,] <- forecast(ets(our_train_set, "ANN"),h=h)$mean
}

#MAE for each horizon
colMeans(abs(medium_noise_holdout - medium_noise_forecasts))



###SES###

#Fit SES with fixed intial seed 
es_ANN_initial_1 <- es(medium_noise, model = "ANN", initial = medium_noise[1],
                       h=h, holdout = TRUE)

es_ANN_initial_1$accuracy

#Fit SES with optimized seed
es_ANN_opt <- es(medium_noise, model = "ANN", h=h, holdout= TRUE)

es_ANN_opt$accuracy

#Benchmarking
#Fit SES with optimized seed 

medium_noise_naive <- es(medium_noise, model = "ANN", persistance = 1, 
                         h=h, holdout = TRUE)

medium_noise_naive$accuracy


##Other SES methods, Holt's method

trend_data <- read.csv("trend_data.csv")

plot(trend_data$x, type = "l")

trend_data <- ts(trend_data, frequency = 12)

plot(trend_data)

trend_data_length <- length(trend_data)
#Split into training and testing
trend_data_train <- ts(trend_data[1:36], frequency = 12)
trend_data_test <- trend_data[37:trend_data_length]

#Calculate Holt Method
ets_ANN <- ets(trend_data_train, model = "AAN")
ets_ANN

coef(ets_ANN)

forecast(ets_ANN, h=h)$mean

plot(forecast(ets_ANN, h=h))

#Calculate a Damped Holt Method 
ets_AAdn <- ets(trend_data_train, model = "AAN", damped = TRUE)

ets_AAdn

#Fit a holt's method , no damped trend
ets(trend_data_train, model = "AAN", damped = FALSE)


es_AAdn <- es(trend_data, model = "AAdN", h=h, holdout = TRUE)


##Holt-Winters

trend_seasonal_data <- read.csv("trend_seasonal_data.csv", header = FALSE)

trend_seasonal_data <- ts(trend_seasonal_data, frequency = 12)
plot(trend_seasonal_data)

trend_seasonal_data_train <- ts(trend_seasonal_data[1:36], frequency = 12)
trend_seasonal_data_test <- trend_seasonal_data[37:trend_data_length]


#Fit a model using ets()
ets_AAA <- ets(trend_seasonal_data_train, model = "AAA", damped = FALSE)
#do the same thing using es():
es_AAA <- es(trend_seasonal_data_train, model = "AAA", h=h)

ets_AAA
es_AAA



#Selecting best model based on optimization

#calculate an optimized ETS method using ets()
ets_ZZZ <- ets(trend_seasonal_data_test, model = "ZZZ")
#Do the same thing using es()
es_ZZZ <- es(trend_seasonal_data_train, model = "ZZZ")

#Select the most appropriate non-seasonal model with ets()
ets_ZZN <- ets(trend_data_train, model = "ZZN")
#Do the same thing with es()
es_ZZN <- es(trend_data_train, model = "ZZN", silent = "a")
