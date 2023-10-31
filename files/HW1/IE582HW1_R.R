### Task 4.0 : Loading required libraries and data.

library(ggplot2)
library(caret)
library(data.table)
library(zoo)
library(GGally)
library(gridExtra)


long <- data.table(read.csv("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/all_ticks_long.csv"))

wide <- data.table(read.csv("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/all_ticks_wide.csv"))

columns_wide <- colnames(wide)

dim(wide)


### Task 4.1 : Descriptive Analysis

str(wide)

wide$timestamp <- as.POSIXct(wide$timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
class(wide$timestamp)

summary(wide)


cat("Value of 368th AEFES record before NA Handling -> ",wide$AEFES[368],"\n")


selected_columns <- wide[0:length(wide),2:61]
colnames(selected_columns)
window_size <- 5

# Apply na.approx() to fill missing values in each column
for (columnName in colnames(selected_columns)) {
  if(any(is.na(wide[[columnName]]))){
    wide[[columnName]] <- na.approx(wide[[columnName]], method = "linear", rule = window(window_size))
  }
}

cat("Value of 368th AEFES record after NA Handling -> ",wide$AEFES[368],"\n")


colSums(is.na(wide))


anomaly_threshold_aefes <-  mean(wide$AEFES) - 3 * sqrt(var(wide$AEFES)) # Adjust this threshold as needed
anomaly_values_aefes <- wide$AEFES < anomaly_threshold_aefes

# Create a line chart with ggplot2 and highlight anomaly values
ggplot(wide, aes(x = timestamp, y = AEFES)) +
  geom_line() +
  geom_point(data = wide[anomaly_values_aefes, ], aes(color = "Anomaly"), size = 3) +
  labs(x = "Datetime", y = "Stock Closing Prices", title = "AEFES Stock Line Chart with Anomalies") +
  scale_color_manual(values = c("Anomaly" = "red")) +
  theme_minimal()



anomaly_threshold_ccola <-  mean(wide$CCOLA) - 3 * sqrt(var(wide$CCOLA)) # Adjust this threshold as needed
anomaly_values_ccola <- wide$CCOLA < anomaly_threshold_ccola

# Create a line chart with ggplot2 and highlight anomaly values
ggplot(wide, aes(x = timestamp, y = CCOLA)) +
  geom_line() +
  geom_point(data = wide[anomaly_values_ccola, ], aes(color = "Anomaly"), size = 3) +
  labs(x = "Datetime", y = "Stock Closing Prices", title = "CCOLA Stock Line Chart with Anomalies") +
  scale_color_manual(values = c("Anomaly" = "purple")) +
  theme_minimal()


selectedStockPairs <- c("AEFES","AKSA","CCOLA","SISE","KCHOL")

isAnomaly <- function(columnName, index){
  
  flag <- FALSE
  
  currentCellValue <- wide[[columnName]][index]
  
  thresholdValue <- mean(wide[[columnName]]) - 3 * sqrt(var(wide[[columnName]]))
  
  if (currentCellValue < thresholdValue || currentCellValue < 0.5) {
    flag <- TRUE
  }
  return(flag)
}

for (columnName in selectedStockPairs){
  for (ind in 1:nrow(wide)){
    if(isAnomaly(columnName,ind)){
      cat("Old value in ",columnName," -> Index: ",ind," -> ", "Value: ", wide[[columnName]][ind],"\n")
      wide[[columnName]][ind] <- round(mean(wide[[columnName]][ind - 1: ind + 1], na.rm = TRUE), digits = 4)
      cat("New value in ",columnName," -> Index: ",ind," -> ", "Value: ", wide[[columnName]][ind],"\n")
    }
  }
}

plot(wide$timestamp, wide$AEFES, type = "l", xlab = "Timestamp", ylab = "Stock Prices", main = "AEFES Stock Prices vs. Time")
plot(wide$timestamp, wide$AKSA, type = "l", xlab = "Timestamp", ylab = "Stock Prices",col = "blue", main = "AKSA Stock Prices vs. Time")
plot(wide$timestamp, wide$CCOLA, type = "l", xlab = "Timestamp", ylab = "Stock Prices",col = "navyblue", main = "CCOLA Stock Prices vs. Time")
plot(wide$timestamp, wide$SISE, type = "l", xlab = "Timestamp", ylab = "Stock Prices",col = "lightblue", main = "SISE Stock Prices vs. Time")
plot(wide$timestamp, wide$KCHOL, type = "l", xlab = "Timestamp", ylab = "Stock Prices",col = "lightblue3", main = "KCHOL Stock Prices vs. Time")


print("Descriptive summary analysis")
for (col in selectedStockPairs){
  cat("Column Name: ", col, "\n")
  cat("Minimum Price: ", min(wide[[col]]), "\n")
  cat("Maximum Price: ", max(wide[[col]]), "\n")
  cat("Price Range: ",max(wide[[col]]) -  min(wide[[col]]), "\n")
  cat("Median Price: ", median(wide[[col]]), "\n")
  cat("Average Price: ", mean(wide[[col]]), "\n")
  cat("Stdev of Price: ", sd(wide[[col]]), "\n")
  cat("-------------------","\n")
  
}

hist(wide$AEFES,breaks = 20, main = "Histogram of AEFES")
hist(wide$AKSA, breaks = 20, main = "Histogram of AKSA")
hist(log(wide$AKSA), breaks = 20, main = "Histogram of log(AKSA)")
hist(wide$CCOLA,breaks = 20, main = "Histogram of CCOLA")
hist(log(wide$CCOLA), breaks = 20, main = "Histogram of log(CCOLA)")
plot(x=wide$timestamp,y=log(wide$CCOLA),type="l")



wide_temp <- wide[,c("timestamp","AEFES","AKSA","CCOLA","SISE","KCHOL")]

wide_combined <- data.frame( "AEFES" = wide_temp[,"AEFES"],
                             "AKSA" = wide_temp[,"AKSA"],
                             "CCOLA" = wide_temp[,"CCOLA"],
                             "SISE" = wide_temp[,"SISE"],
                             "KCHOL" = wide_temp[,"KCHOL"])
matplot(x = wide_temp$timestamp,y = wide_combined, type = "l",pch=1,col = 1:5,
        xlab = "Time", ylab = "Stock Prices")
legend("topright", legend = c("AEFES","AKSA","CCOLA","SISE","KCHOL"), col=1:5,pch=0.2,title = "Stock", cex = 0.5)

### Task 4.2 : Moving Window Correlation


wide_filtered <- wide[,c("timestamp","AEFES","AKSA","CCOLA","SISE","KCHOL")]

wide_filtered$Year <- format(wide_filtered$timestamp, format = "%Y")
wide_filtered$Month <- month.name[as.numeric(format(wide_filtered$timestamp, format = "%m"))]

wide_filtered <- wide_filtered[, c("timestamp","Year","Month","AEFES","AKSA","CCOLA","SISE","KCHOL")]
head(wide_filtered)


cor(wide_filtered[,c("AEFES","AKSA","CCOLA","SISE","KCHOL")])

ggcorr(wide_filtered,
       method = c("pairwise"),
       nbreaks = 6,
       hjust = 0.8,
       label = TRUE,
       label_size = 3,
       color = "grey20")

plot(~ AEFES + AKSA + CCOLA + SISE + KCHOL, data = wide_filtered, main = "Stock Price's Correlation")


monthList <- unique(wide_filtered$Month)
yearList <- unique(wide_filtered$Year)

movingWindowCorr <- function(windowType,uniqueList){
  
  if(windowType == "Month"){
    for (month in uniqueList){
      filt <- subset(wide_filtered, Month == month)
      #cat(month,": ", round(cor(filt$SISE,filt$KCHOL),digits=4), "\n" )
      cat("*-------  ",month,"\n")
      print(cor(filt[,c("AEFES","AKSA","CCOLA","SISE","KCHOL")]))
      print("*------------------------------*")
      # plotCorr <-ggcorr(filt,
      #                   method = c("pairwise"),
      #                   nbreaks = 6,
      #                   hjust = 0.8,
      #                   label = TRUE,
      #                   label_size = 3,
      #                   label_round = 3,
      #                   color = "grey20")
      # print(plotCorr)
    } 
  } else if (windowType == "Year"){
    
    for (year in uniqueList){
      filt <- subset(wide_filtered, Year == year)
      cat("*-------  ",year,"\n")
      print(cor(filt[,c("AEFES","AKSA","CCOLA","SISE","KCHOL")]))
      print("*------------------------------*")
      # plotCorr <-ggcorr(filt,
      #                   method = c("pairwise"),
      #                   nbreaks = 6,
      #                   hjust = 0.8,
      #                   label = TRUE,
      #                   label_size = 3,
      #                   label_round = 3,
      #                   color = "grey20")
      # print(plotCorr)
    }
  }
  
}

movingWindowCorr("Month",monthList)
movingWindowCorr("Year",yearList)


corrByMonthAEFES_KCHOL <- c()
corrByMonthCCOLA_AEFES <- c()
corrByMonthKCHOL_AKSA <- c()
corrByMonthSISE_KCHOL <- c()
i <- 1
for (month in monthList){
  
  month_df <- subset(wide_filtered, Month == month)
  corrByMonthAEFES_KCHOL[i] <- round(cor(month_df$AEFES,month_df$KCHOL),digits=2)
  corrByMonthCCOLA_AEFES[i] <- round(cor(month_df$CCOLA,month_df$AEFES),digits=2)
  corrByMonthKCHOL_AKSA[i] <- round(cor(month_df$KCHOL,month_df$AKSA),digits=2)
  corrByMonthSISE_KCHOL[i] <- round(cor(month_df$SISE,month_df$KCHOL),digits=2)
  i <- i + 1
  
}


monthCorrData <- data.frame("Month" = unique(wide_filtered$Month), "AEFES_KCHOL" = corrByMonthAEFES_KCHOL
                            ,"CCOLA_AEFES" = corrByMonthCCOLA_AEFES, "KCHOL_AKSA" = corrByMonthKCHOL_AKSA,
                            "SISE_KCHOL" = corrByMonthSISE_KCHOL)

monthCorrDataCombined <- data.frame( "AEFES_KCHOL" = monthCorrData[,"AEFES_KCHOL"],
                                     "CCOLA_AEFES" = monthCorrData[,"CCOLA_AEFES"],
                                     "KCHOL_AKSA" = monthCorrData[,"KCHOL_AKSA"],
                                     "SISE_KCHOL" = monthCorrData[,"SISE_KCHOL"])
matplot(y = monthCorrDataCombined, type = "l",pch=1,col = 1:4,
        xlab = "Time", ylab = "Correlation")
legend("right", legend = c("AEFES_KCHOL","CCOLA_AEFES","KCHOL_AKSA","SISE_KCHOL"),
       col=1:4,pch=0.2,cex = 0.5)


corrByYearAEFES_KCHOL <- c()
corrByYearCCOLA_AEFES <- c()
corrByYearKCHOL_AKSA <- c()
corrByYearSISE_KCHOL <- c()
i <- 1
for (year in yearList){
  
  year_df <- subset(wide_filtered, Year == year)
  corrByYearAEFES_KCHOL[i] <- round(cor(year_df$AEFES,year_df$KCHOL),digits=2)
  corrByYearCCOLA_AEFES[i] <- round(cor(year_df$CCOLA,year_df$AEFES),digits=2)
  corrByYearKCHOL_AKSA[i] <- round(cor(year_df$KCHOL,year_df$AKSA),digits=2)
  corrByYearSISE_KCHOL[i] <- round(cor(year_df$SISE,year_df$KCHOL),digits=2)
  i <- i + 1
  
}


yearCorrData <- data.frame("Year" = unique(wide_filtered$Year), "AEFES_KCHOL" = corrByYearAEFES_KCHOL
                           ,"CCOLA_AEFES" = corrByYearCCOLA_AEFES, "KCHOL_AKSA" = corrByYearKCHOL_AKSA,
                           "SISE_KCHOL" = corrByYearSISE_KCHOL)

yearCorrDataCombined <- data.frame( "AEFES_KCHOL" = yearCorrData[,"AEFES_KCHOL"],
                                    "CCOLA_AEFES" = yearCorrData[,"CCOLA_AEFES"],
                                    "KCHOL_AKSA" = yearCorrData[,"KCHOL_AKSA"],
                                    "SISE_KCHOL" = yearCorrData[,"SISE_KCHOL"])
matplot(x=yearCorrData$Year,y = yearCorrData[,-1], type = "l",pch=1,col = 1:4,
        xlab = "Time", ylab = "Correlation")
legend("bottomright", legend = c("AEFES_KCHOL","CCOLA_AEFES","KCHOL_AKSA","SISE_KCHOL"),
       col=1:4,pch=0.2,cex = 0.5)

### Task 4.3 : Principal Component Analysis (PCA)

pcaObj_all <- princomp(wide[,-1],cor=T)

summary(pcaObj_all)

plot(pcaObj_all)


pcaObj <- princomp(wide_filtered[,-1:-3],cor=T)
summary(pcaObj, loadings = T)
biplot(pcaObj, scale = 0)
plot(pcaObj)


logPcaObj <- princomp(log(wide_filtered[,-1:-3]),cor=T)
summary(logPcaObj, loadings = T)
biplot(logPcaObj, scale = 0)
plot(logPcaObj)


### Task 4.4 : Inference with Google Trends


google_aefes <- 
  fread("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/GoogleTrend/multiTimelineAEFES.csv")
google_aksa <- 
  fread("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/GoogleTrend/multiTimelineAKSA.csv")
google_ccola <- 
  fread("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/GoogleTrend/multiTimelineCCOLA.csv")
google_sise <- 
  fread("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/GoogleTrend/multiTimelineSISE.csv")
google_kchol <- 
  fread("C:/Users/anil.turgut/Desktop/IE582/HW1/Dataset/GoogleTrend/multiTimelineKCHOL.csv")

df_google <- data.frame("Time"=google_aefes[,1], "AEFES"=google_aefes[,-1], "AKSA"=google_aksa[,-1],
                        "CCOLA"=google_ccola[,-1],"SISE"=google_sise[,-1], "KCHOL"=google_kchol[,-1])

colnames(df_google) <- c("Month","AEFES","AKSA","CCOLA","SISE","KCHOL")

head(df_google)


for (ind in 1:nrow(df_google)){
  df_google[["Year"]][ind] <- strsplit(df_google[["Month"]][ind], "-")[[1]][1]
}

df_google <- df_google[, c("Month","Year","AEFES","AKSA","CCOLA","SISE","KCHOL")]
df_google_month <- df_google[, c("Month","AEFES","AKSA","CCOLA","SISE","KCHOL")]

df_google_month


# Line chart of 5 stocks in google trend
df_google_combined <- data.table::melt(df_google_month, id.var = "Month")


ggplot(df_google_combined, aes(x = Month, y = value, color = variable, group=variable)) +
  geom_line() +
  theme_minimal()


corrGoogleByYearAEFES_KCHOL <- c()
corrGoogleByYearCCOLA_AEFES <- c()
corrGoogleByYearKCHOL_AKSA <- c()
corrGoogleByYearSISE_KCHOL <- c()
i <- 1
for (year in unique(df_google$Year)){
  
  year_df <- subset(df_google, Year == year)
  corrGoogleByYearAEFES_KCHOL[i] <- round(cor(year_df$AEFES,year_df$KCHOL),digits=2)
  corrGoogleByYearCCOLA_AEFES[i] <- round(cor(year_df$CCOLA,year_df$AEFES),digits=2)
  corrGoogleByYearKCHOL_AKSA[i] <- round(cor(year_df$KCHOL,year_df$AKSA),digits=2)
  corrGoogleByYearSISE_KCHOL[i] <- round(cor(year_df$SISE,year_df$KCHOL),digits=2)
  i <- i + 1
  
}

googleYearCorrData <- data.frame("Year" = unique(df_google$Year), "AEFES_KCHOL" = corrGoogleByYearAEFES_KCHOL
                                 ,"CCOLA_AEFES" = corrGoogleByYearCCOLA_AEFES, "KCHOL_AKSA" = corrGoogleByYearKCHOL_AKSA,
                                 "SISE_KCHOL" = corrGoogleByYearSISE_KCHOL)
googleYearCorrData

matplot(x=googleYearCorrData$Year,y = googleYearCorrData[,-1], type = "l",pch=1,col = 1:4,
        xlab = "Time", ylab = "Correlation")
legend("topright", legend = c("AEFES_KCHOL","CCOLA_AEFES","KCHOL_AKSA","SISE_KCHOL"),
       col=1:4,pch=0.2,cex = 0.5)


matplot(x=yearCorrData$Year,y = yearCorrData[,-1], type = "l",pch=1,col = 1:4,
        xlab = "Time", ylab = "Correlation")
legend("bottomright", legend = c("AEFES_KCHOL","CCOLA_AEFES","KCHOL_AKSA","SISE_KCHOL"),
       col=1:4,pch=0.2,cex = 0.5)
