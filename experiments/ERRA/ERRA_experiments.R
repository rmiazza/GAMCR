
rm(list=ls())  #clear environment

library(data.table)
library(dplyr)
#install.packages("lubridate")


source("./ERRA_v1.0.R")  # ensemble rainfall runoff analysis script

allGISIDs = c('44', '48', '50', '58', '88','112')
all_h = c(1,2,2,2,1,2)
all_m = c(50,25,70,50,50,50)

all_EZG = c(58.7,125.7,47.5, 32.4,34.3,185.1)
all_modes = c("streamflow","training","complete")
for (ite in 1:6){ 
    GISID = allGISIDs[ite]
    h = all_h[ite]
    m = all_m[ite]
    for (mode in all_modes){
        dat <- read.csv(sprintf("../RES_GAMCR/real_data_paper_v2_seasonal/%s/data_%s.txt", GISID, GISID), header=TRUE, sep=",")    #this is the data file
        dat$date <- as.POSIXct(dat$date, format = "%Y-%m-%d %H:%M:%S")

        # Filter Rows by column value
        
        init = 24*30*16+24*10
        dat$year <- format(dat$date, "%Y")  # Extracts year
        dat$year <- as.numeric(dat$year)   # Converts to numeric


        if (mode == "training"){
            dat <- filter(dat, year < 2018)
        }    
        
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
        #dat_filtered <- dat %>% filter(month > 5 & month < 11)
        
        
        #-----------------------------------------------------------------------------
        folder_path <- sprintf("./output_ERRA_forGAMCR/GISID-%s", GISID)

        # Create folder if it does not exist
        if (!file.exists(folder_path)) {
          dir.create(folder_path, recursive = TRUE)
        }
        fileID <- sprintf("./output_ERRA_forGAMCR/GISID-%s/%s", GISID, mode)
        
        p <- dat$p[init:length(dat$p)]
        q <- dat$q[init:length(dat$q)] 

        q <- q * 3600 * 1000 / (all_EZG[ite] * 1000000)
        
        #------------------------------------------------------------------------------------------------------------------------------
        #snow free season values:
        if (mode == "streamflow"){
            if (GISID == '48'){
                zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=c(5,40), xknot_type="even", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))
            }else{
                 zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=c(5,40), xknot_type="even", robust=FALSE, show_top_xknot = TRUE, Qfilter=ifelse((dat$year[init:length(dat$p)]<=2017), 1, 0))
            }
            with(zz, {                                          # write the output to files
              fwrite(Qcomp, paste0(fileID, "_Qcomp_", options, ".txt"), sep="\t")
            })
        } else { if (GISID == '48'){
            
            zz <- ERRA(p=p, q=q, m=m, h=h, agg=2, xknots=c(5,40), xknot_type="even", robust=FALSE, show_top_xknot = TRUE, Qfilter=snowfree) 
            }else{
                zz <- ERRA(p=p, q=q, m=m, h=h, xknots=c(5,40), xknot_type="even", robust=FALSE, show_top_xknot = TRUE, Qfilter=snowfree) 
    
                }
          with(zz, {                                          # write the output to files
          fwrite(peakstats, paste0(fileID, "_peakstats_", options, ".txt"), sep="\t") 
          fwrite(wtd_avg_RRD, paste0(fileID, "_avgRRD_", options, ".txt"), sep="\t") 
          fwrite(NRF, paste0(fileID, "_NRF_", options, ".txt"), sep="\t") 
            })
        }
    
        
        
    }
}

