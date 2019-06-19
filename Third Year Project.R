library(wordcloud)
library(gmodels)
library(tm)
library(tidyverse)
library(plyr)
library(e1071)

#reading the data
Sys.setlocale('LC_ALL','C')
data.frame <-read.csv(file.choose())
data.frame <- data.frame[-1]
dim(data.frame)
summary(data.frame)
head(data.frame)
str(data.frame)

undesired_words <- c("just", "get","think", "one", "time", "back", "will", "want", "even", "look", "youre")

head(data.frame)

#Cleaning the data
head(data.frame)
data.frame$Sentiment <- factor(data.frame$Sentiment, levels = c(0, 1),
                               labels = c("Negative","Positive"))
head(data.frame)

data.frame$SentimentText <- as.character(data.frame$SentimentText)
head(data.frame)
str(data.frame)

corpus <- Corpus(VectorSource(data.frame$SentimentText))
inspect(corpus[414])

urlPat<-function(x) gsub("(ftp|http)(s?)://.*\\b", "", x)
tun<-function(x) gsub("[@][a - zA - Z0 - 9_]{1,15}", "", x)
tt<-function(x) gsub("RT |via", "", x)
emlPat<-function(x) gsub("\\b[A-Z a-z 0-9._ - ]*[@](.*?)[.]{1,3} \\b", "", x)
udfunction <- filter(undesired_words)

clean.corpus <- tm_map(corpus, removePunctuation)
clean.corpus <- tm_map(clean.corpus, stripWhitespace)
clean.corpus<-tm_map(clean.corpus, urlPat)
clean.corpus <-tm_map(clean.corpus, tun)
clean.corpus <-tm_map(clean.corpus, tt)
clean.corpus <-tm_map(clean.corpus, emlPat)
clean.corpus <- tm_map(clean.corpus, removePunctuation)
clean.corpus <- tm_map(clean.corpus, tolower)
clean.corpus <- tm_map(clean.corpus, removeWords, stopwords())
clean.corpus <- tm_map(clean.corpus, removeNumbers)
clean.corpus <- tm_map(clean.corpus, udfunction)

clean.corpus.dtm <- DocumentTermMatrix(clean.corpus)
inspect(clean.corpus.dtm)


#Splitting the data
n <- nrow(data.frame)
raw.text.train <- data.frame[1:round(.8*n),]
raw.text.test <- data.frame[(round(.8*n)+1):n,]
raw.test.text

nn <- length(clean.corpus)
clean.corpus.train <- clean.corpus[1:round(.8 * nn)]
clean.corpus.test  <- clean.corpus[(round(.8 * nn)+1):nn]

nnn <- nrow(clean.corpus.dtm)
clean.corpus.dtm.train <- clean.corpus.dtm[1:round(.8 * nnn),]
clean.corpus.dtm.test  <- clean.corpus.dtm[(round(.8 * nnn)+1):nnn,]


#removing frequent words
freq.terms <- findFreqTerms(clean.corpus.dtm.train, 30)
clean.corpus.dtm.freq.train <- DocumentTermMatrix(clean.corpus.train, list(dictionary = freq.terms))
clean.corpus.dtm.freq.test  <- DocumentTermMatrix(clean.corpus.test, list(dictionary = freq.terms))


convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

clean.corpus.dtm.freq.train

raw.text.train
#More cleaning
clean.corpus.dtm.freq.train <- apply(clean.corpus.dtm.freq.train, MARGIN = 2, convert_counts)
clean.corpus.dtm.freq.test  <- apply(clean.corpus.dtm.freq.test, MARGIN = 2, convert_counts)

clean.corpus.dtm.freq.train
#Crearing and testing the model
text.classifer <- naiveBayes(clean.corpus.dtm.freq.train, raw.text.train$Sentiment, lapace =1)
text.pred <- predict(text.classifer, clean.corpus.dtm.freq.test)

CrossTable(text.pred, raw.text.test$Sentiment,
           prop.chisq = FALSE, 
           prop.t = FALSE,
           dnn = c('predicted', 'actual'))

head(raw.text.test$Sentiment)
