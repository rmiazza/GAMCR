
rm(list=ls())  #clear environment

library(data.table)
library(dplyr)
#install.packages("lubridate")


source("./ERRA_v1.0.R")  # ensemble rainfall runoff analysis script

output_folder = "../real_data/data_and_visualization"

allGISIDs = c('44', '48', '50', '58', '88','112')
all_h = c(1,2,2,2,1,2)
all_m = c(50,25,70,50,50,50)

#all_EZG = c(58.7,125.7,47.5, 32.4,34.3,185.1)
all_modes = c("streamflow","training","complete")
for (ite in 1:6){ 
    GISID = allGISIDs[ite]
    h = all_h[ite]
    m = all_m[ite]
    for (mode in all_modes){
        dat <- read.csv(sprintf("../real_data/%s/data_%s.txt", GISID, GISID), header=TRUE, sep=",")    #this is the data file
        dat$date <- as.POSIXct(dat$date, format = "%Y-%m-%d %H:%M:%S")

        # Filter Rows by column value
        
        
        # The init parameter should be specified depending on the Tmax value and the set of features chosen to train GAMCR. Indeed, to compare ERRA and GAMCR results, we want to make sure both models have be trained on exactly the same dataset. But since GAMCR is a feature-based model, one needs some data history before the first training time point. Here init is thus the sum of the maximum window used to compute features in GAMCR and the Tmax value. 
        init = 24*30*16+24*10
        dat$year <- format(dat$date, "%Y")  # Extracts year
        dat$year <- as.numeric(dat$year)   # Converts to numeric


        if (mode == "training"){
            dat <- filter(dat, year < 2018)
        }    
        
        # We keep only the snowfree period
        if (GISID == '46'){
            low_month = 7
            up_month = 9
        }else{    if (GISID %in% c('3','44','112')){
                    low_month = 6
                    up_month = 10
                 }else{ low_month = 5
                        up_month = 10
                      }
             }
        dat$month <- format(dat$date, "%m")  # Extracts month as character ("01", "02", etc.)
        dat$month <- as.numeric(dat$month)   # Converts to numeric (1, 2, etc.) if needed

        snowfree <- ifelse(dat$month[init:length(dat$month)] >= low_month & 
                   dat$month[init:length(dat$month)] <= up_month, 1, 0)
        # Filter the dataframe using dplyr      
        
        #-----------------------------------------------------------------------------
        folder_path <- sprintf("%s/output_ERRA_forGAMCR/GISID-%s", output_folder, GISID)

        # Create folder if it does not exist
        if (!file.exists(folder_path)) {
          dir.create(folder_path, recursive = TRUE)
        }
        fileID <-  sprintf("%s/output_ERRA_forGAMCR/GISID-%s/%s", output_folder, GISID, mode)
        
        p <- dat$p[init:length(dat$p)]
        p[p<0.5] <- 0
        q <- dat$q[init:length(dat$q)]


        levels = c(10,20,30,40,50,60,70,80,90,95,97.5)
        p4quantiles = p
        p4quantiles[snowfree==FALSE] <- 0
        p4quantiles <- sort(p4quantiles[p4quantiles>0])
        quantiles <- quantile(p4quantiles, probs=levels/100, na.rm=TRUE)  
        print(GISID)
        print(quantiles)
        
        #------------------------------------------------------------------------------------------------------------------------------
        #snow free season values:
        if (mode == "streamflow"){
            if (GISID == '48'){
                #zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=c(5,40), xknot_type="even", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))
                #zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=c(5,10,15,20,25,30,35,40,45,50,60,65,70,75,80,85,90,95), xknot_type="percentiles", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))       
                zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=quantiles, xknot_type="values", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))
            }else{
                zz <- ERRA(p=p, q=q, m=m, h=h, xknots=quantiles, xknot_type="values", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))
            }
            with(zz, {                                          # write the output to files
              fwrite(Qcomp, paste0(fileID, "_Qcomp_", options, ".txt"), sep="\t")
            })
        } else { if (GISID == '48'){
            zz <- ERRA(p=p, q=q, m=m, h=h, xknots=quantiles, xknot_type="values", robust=FALSE, show_top_xknot = TRUE, Qfilter=snowfree)

            }else{
                zz <- ERRA(p=p, q=q, m=m, h=h, xknots=quantiles, xknot_type="values", robust=FALSE, show_top_xknot = TRUE, Qfilter=snowfree)
                }
          with(zz, {                                          # write the output to files
          fwrite(peakstats, paste0(fileID, "_peakstats_", options, ".txt"), sep="\t") 
          fwrite(wtd_avg_RRD, paste0(fileID, "_avgRRD_", options, ".txt"), sep="\t") 
          fwrite(NRF, paste0(fileID, "_NRF_", options, ".txt"), sep="\t") 
            })
        }
    }
}