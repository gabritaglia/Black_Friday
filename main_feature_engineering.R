# Test RMSE: 2462.1137687
#
# @author: BTData
library(dplyr)
library(reshape2)
library(caret)
library(data.table) # Versione development
library(Laurae) # Versione patchata https://github.com/manuel-calzolari/Laurae
library(MLmetrics)

set.seed(0) # Fissa seed per essere riproducibile


preprocess_data <- function() {
    # Legge i file
    df_train <- read.csv("train.csv", stringsAsFactors = FALSE)
    df_test <- read.csv("test.csv", stringsAsFactors = FALSE)

    # Concatena i file per poterli pre-processare congiuntamente in modo semplice
    df <- bind_rows(df_train, df_test)

    # Mette da parte gli ID originali del test set (servono per la submission)
    df_test_ids <- df_test[, c("User_ID", "Product_ID")]

    # Esegue il label encoding degli ID, del genere, dell'occupazione e della categoria di citta'
    # (successivamente verranno considerate colonne categoriche)
    df$User_ID <- as.integer(as.factor(df$User_ID)) - 1
    df$Product_ID <- as.integer(as.factor(df$Product_ID)) - 1
    df$Gender <- as.integer(as.factor(df$Gender)) - 1
    df$Occupation <- as.integer(as.factor(df$Occupation)) - 1
    df$City_Category <- as.integer(as.factor(df$City_Category)) - 1

    # Prende la media del range di eta' come valore numerico (considera 55+ come 55-80)
    df$Age[df$Age == "55+"] <- "55-80"
    df$Age <- sapply(df$Age, function(x) mean(as.numeric(strsplit(x, "-")[[1]])))

    # Rimpiazza 4+ con 4 e converte in valore numerico
    df$Stay_In_Current_City_Years[df$Stay_In_Current_City_Years == "4+"] <- "4"
    df$Stay_In_Current_City_Years <- as.numeric(df$Stay_In_Current_City_Years)

    
    # Dummy per le categorie dei prodotti
    cats_labels <- c("Product_Category_1", "Product_Category_2", "Product_Category_3")
    df$id <- seq.int(nrow(df))
    cats <- recast(df[, c("id", cats_labels)], id ~ value, id.var = "id", length)
    cats[, c("id", "NA")] <- NULL
    colnames(cats) <- paste("PC", colnames(cats), sep = "_")
    # df[, c("id", cats_labels)] <- NULL #questo comando lo eseguo dopo il set di istruzioni per creare le variabili basate su Purchase
    df <- bind_cols(df, cats)
    
    #nota queste variabili le calcolo prima della ricodifica della categoria in dummies, ma le aggiungo dopo per non scombinare l'ordinamento
    #User
    #Aggiungo il  count, purchase medio, mediano, q1, q3, max, min per User_ID
    df_User <- df %>% 
                  select(User_ID, Purchase) %>%
                  group_by(User_ID) %>%
                  summarise(User_Count = n(),
                            Mean_Purchase_User = mean(Purchase, na.rm=TRUE),
                            Median_Purchase_User = median(Purchase, na.rm=TRUE),
                            Q1_Purchase_User = quantile(Purchase,0.25,na.rm=TRUE),
                            Q3_Purchase_User = quantile(Purchase,0.75,na.rm=TRUE),
                            min_Purchase_User = min(Purchase,na.rm=TRUE),
                            max_Purchase_User = max(Purchase,na.rm=TRUE),
                            sd_Purchase_User = sd(Purchase,na.rm=TRUE)
                            )
    df <- left_join( df, df_User, by = "User_ID")  
        #Product
    Product_ID_test<-df %>% filter(is.na(Purchase)) %>% distinct(Product_ID, Product_Category_1)
    #Aggiungo il  count, purchase medio, mediano, q1, q3, max, min per Product_ID solo per i Product_ID nel che sono nel train
    df_Product_train <- df %>% 
                          select(Product_ID,Product_Category_1,Purchase) %>%
                          filter(Product_ID %in% Product_ID_test$Product_ID) %>%
                          group_by(Product_ID,Product_Category_1) %>%
                          summarise(Product_Count = n(),
                                    Mean_Purchase_Product = mean(Purchase, na.rm=TRUE),
                                    Median_Purchase_Product = median(Purchase, na.rm=TRUE),
                                    Q1_Purchase_Product = quantile(Purchase,0.25,na.rm=TRUE),
                                    Q3_Purchase_Product = quantile(Purchase,0.75,na.rm=TRUE),
                                    min_Purchase_Product = min(Purchase,na.rm=TRUE),
                                    max_Purchase_Product = max(Purchase,na.rm=TRUE),
                                    sd_Purchase_Product = sd(Purchase,na.rm=TRUE))
                    
    #creo un dataframe provvisorio con i Product_ID non presenti nel train e il relativo count
    temp_Product_test <- df %>% 
                            select(Product_ID, Purchase, Product_Category_1) %>%
                            filter(!(Product_ID %in% Product_ID_test$Product_ID)) %>%
                            group_by(Product_ID,Product_Category_1) %>%
                            summarise(Product_Count = n())
    #solo per i Product_ID non presenti nel train associo il purchase medio, mediano, q1, q3, max, min di categoria
    temp_Product_NA <- df_Product_train %>% 
                                        group_by(Product_Category_1) %>%
                                        summarise(Mean_Purchase_Product = median(Mean_Purchase_Product, na.rm=TRUE),
                                                  Median_Purchase_Product = median(Median_Purchase_Product, na.rm=TRUE),
                                                  Q1_Purchase_Product = median(Q1_Purchase_Product,na.rm=TRUE),
                                                  Q3_Purchase_Product = median(Q3_Purchase_Product,na.rm=TRUE),
                                                  min_Purchase_Product = median(min_Purchase_Product,na.rm=TRUE),
                                                  max_Purchase_Product = median(max_Purchase_Product,na.rm=TRUE),
                                                  sd_Purchase_Product = median(sd_Purchase_Product,na.rm=TRUE))
    #Metto insieme i due dataset provvisori relativi ai Product_ID presenti nel solo test set
    df_Product_test <- left_join(temp_Product_test, temp_Product_NA, by = "Product_Category_1") 
    #combino i due dataset prodotto del train e del test e ne faccio la join con il df originario
    df_Product <- bind_rows(df_Product_train, df_Product_test) %>% select(-Product_Category_1)
    #Aggiungo al dataset le variabili relative alle statistiche su Purchase per Product_ID che ho calcolato prima
    df <- left_join( df, df_Product, by = "Product_ID")  
    
    #Elimino le variabili id, Product_Category_1, Product_Category_2, Product_Category_3 dal dataset che ora non servono più
    df[, c("id", cats_labels)] <- NULL
    # Sposta la colonna Purchase alla fine
    df <- df %>% select( - Purchase, everything())

    # Ridivide il train set dal test set
    br <- nrow(df_train)
    df_train <- head(df, br)
    df_test <- tail(df, - br)

    # Divide X da y
    X_train <- df_train
    X_train$Purchase <- NULL
    y_train <- df_train$Purchase
    X_test <- df_test
    X_test$Purchase <- NULL

    return(list("X_train" = X_train, "y_train" = y_train, "X_test" = X_test, "df_test_ids" = df_test_ids))
}


run <- function(data) {
    # K-fold cross-validation
    rmses_val <- vector('numeric')
    n_folds <- 5
    preds_test <- matrix(nrow = nrow(data[["X_test"]]), ncol = n_folds)
    #NOT shuffle
    # shuffle <- sample(length(data[["y_train"]]))
    # data[["X_train"]] <- data[["X_train"]][shuffle,]
    # data[["y_train"]] <- data[["y_train"]][shuffle]
    kf <- createFolds(data[["y_train"]], k = n_folds)
    for (i in 1:n_folds) {
        X_train <- data[["X_train"]][ - kf[[i]],]
        X_val <- data[["X_train"]][kf[[i]],]
        y_train <- data[["y_train"]][ - kf[[i]]]
        y_val <- data[["y_train"]][kf[[i]]]

        # Gradient Boosting Machine (LightGBM)
        # https://github.com/Microsoft/LightGBM
        # X_val e y_val vengono utilizzati per l'early stopping
        # TODO: ottimizzare iperparametri
        regr <- lgbm.train(lgbm_path = Sys.getenv("LIGHTGBM_EXEC"),
                           workingdir = paste(getwd(), "temp", sep = "/"),
                           application = "regression",
                           num_iterations = 10000,
                           learning_rate = 0.01,
                           num_threads = 4,
                           max_bin = 8191,
                           num_leaves = 1023,
                           feature_fraction = 0.5,
                           min_data_in_leaf = 10,
                           min_sum_hessian_in_leaf = 5.0,
                           x_train = as.data.table(X_train),
                           y_train = y_train,
                           x_val = as.data.table(X_val),
                           y_val = y_val,
                           early_stopping_rounds = 20,
                           categorical_feature = c(0, 1, 2, 4, 5, 7),
                           x_test = as.data.table(data[["X_test"]]),
                           predictions = TRUE)

        # Previsione e accumulo del RMSE per il validation set del fold
        pred_val <- regr$Validation
        rmses_val <- c(rmses_val, RMSE(pred_val, y_val))

        # Previsione e accumulo della previsione per il test set (basata sul fold corrente)
        pred_test <- regr$Testing
        preds_test[, i] <- pred_test
    }

    # Calcola la media dei RMSE per i vari fold
    rmse_val <- mean(rmses_val)

    # Calcola la media delle previsioni basate sui vari fold e la aggiunge al dataframe in output
    pred <- apply(preds_test, 1, mean)
    data[["df_test_ids"]]$Purchase <- pred

    return(list("rmse" = rmse_val, "df_submission" = data[["df_test_ids"]]))
}


main <- function() {
    print("Pre-processing data...")
    data <- preprocess_data()

    print("Processing data...")
    output <- run(data)

    print(paste("Cross-validation RMSE:", output[["rmse"]]))

    print("Generating submission data...")
    write.csv(output[["df_submission"]], "submission_r.csv", row.names = FALSE, quote = FALSE)

    print("Export output data...")
    save(output,file = "output.Rdata")
    
    print("Done.")
}


main()
