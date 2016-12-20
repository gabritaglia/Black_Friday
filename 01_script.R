library(ggplot2)
library(dplyr)

path<-"C:\\Users\\BTData\\Documents\\Projects\\Black Friday"
  
setwd(path)  
dati<-read.csv(file = "train.csv",header = T)

names(dati)
dati$User_ID<-as.factor(dati$User_ID)
dati$Product_Category_1<-as.factor(dati$Product_Category_1)
dati$Product_Category_2<-as.factor(dati$Product_Category_2)
dati$Product_Category_3<-as.factor(dati$Product_Category_3)

summary(dati)


ggplot(dati, aes(x = Age))+geom_bar()
ggplot(dati, aes(x = User_ID))+geom_bar()

ggplot(dati, aes(x = Purchase))+geom_histogram()


dati%>%select(Product_Category_1, Purchase)%>%group_by(Product_Category_1)%>%summarise(mean(Purchase))


length(unique(dati[,c("User_ID","Gender", "Age", "Occupation","City_Category","Stay_In_Current_City_Years","Marital_Status")]))
length(unique(dati[,c("User_ID","Gender", "Age", "Occupation")]))


dati2<-dati%>%select(Age, Occupation,Purchase,City_Category)%>%group_by(Age, Occupation,City_Category)%>%summarise(n=n(), p=mean(Purchase))
ggplot(dati2, aes(Age, Occupation, size=n))+geom_point()
ggplot(dati2, aes(Age, Occupation, size=p))+geom_point()
ggplot(dati2, aes(Age, City_Category, size=p))+geom_point()
