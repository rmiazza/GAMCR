

# this script aims to calibrate the model parameters using time series from a real catchment 

setwd("/home/duchemin/Desktop/data_watres/benchmark/flow/calibration_PSL_simulatedata/dataQuentin/")  # change this to wherever the data file and called scripts are

rm(list=ls())  #clear environment

library(data.table)
library(dplyr)

library(Rcpp)  #load Rcpp

sourceCpp("/home/duchemin/Desktop/data_watres/benchmark/code/threebox_light_v1.0_for_Rcpp.cpp")  #this is the benchmark model code (water flux module, slimmed down for parameter optimization)



RMS <- function(x, wt=rep(1, length(x))) {  # root-mean-square average
  return(sqrt(weighted.mean(x*x, w=wt, na.rm=TRUE)))
}

weighted_se <- function(x, wt=rep(1, length(x))) {  # weighted standard error
  xbar <- weighted.mean(x, wt, na.rm=TRUE)
  std <- sqrt( weighted.mean( (x-xbar)*(x-xbar), wt, na.rm=TRUE) )
  n <- sum(!is.na(x*wt))
  return(std/sqrt(n))
}


#####################################################
# NSE for parameter optimization
#####################################################
NSE <- function(params, p, pet, q, dt, pet_mult) {
  
  input <- list( bu = params[1],  #first eight entries are model parameters
                 bl = params[2],
                 su_ref = params[3],  #upper box reference storage
                 sl_ref = params[4],  #lower box reference storage
                 sc_ref = params[5],  #channel box reference storage
                 fw = params[6],
                 f_OF = params[7],
                 f_SS = params[8],
                 bc = 1.5,            # bc fixed by hydraulic geometry relationships w ~ Q^0.26 and d ~ Q^0.40 implying Q ~ volume^1.5
                 n = length(p),       
                 tol = 0.001,         # tolerance for numerical integration
                 p = p,               # precip time series from input file
                 pet = pet,           # potential ET from input file
                 pet_mult = pet_mult,
                 dt = dt)
  
  mod <- threebox_model_light(input)    # run model
  
  return( 1.0-var((mod$qa-q), na.rm=TRUE)/var(q, na.rm=TRUE) - sum(is.na(mod$qa)))   # includes penalty for NA's
  
}




# Convert POSIXct to fractional year
fractional_year <- function(date) {
  year <- as.numeric(format(date, "%Y"))
  start_of_year <- as.POSIXct(paste0(year, "-01-01 00:00:00"))
  end_of_year <- as.POSIXct(paste0(year + 1, "-01-01 00:00:00"))
  
  # Calculate the fraction of the year that has passed
  year_fraction <- as.numeric(difftime(date, start_of_year, units = "secs")) / 
    as.numeric(difftime(end_of_year, start_of_year, units = "secs"))
  
  return(year + year_fraction)
}


if (FALSE){
  dat <- fread("GISID-81_BAFU-2432.csv", header=TRUE, sep=",", na.strings=c("NA",".","","#N/A"))    #this is the test data file
  
  datetime <- dat$datetime;
  
  t <- fractional_year(datetime)
  
  p <- dat$P_mmhr 
  if (is.null(p)) p <- dat$P_mmhr   # to handle different column headers
  p[is.na(p)] <- 0
  
  pet <- dat$PET_mmhr
  if (is.null(pet)) pet <- dat$PET_mmhr  # to handle different column headers
  NonNAindex <- which(!is.na(pet))
  meanpet <- mean(pet[NonNAindex])
  pet[is.na(pet)] <- meanpet
  
  q <- dat$Q_mmhr
  if (is.null(q)) q <- dat$Q_mmhr   # to handle different column headers
  
  NonNAindex <- which(!is.na(q))
  meanq <- mean(q[NonNAindex])
  
  q[is.na(q)] <- meanq
  
  #  q <- dat$Q_cms
  # if (is.null(q)) q <- dat$Q_cms   # to handle different column headers
}
if (FALSE){
  dat1 <- fread("../../3_long_stations/Sion.csv", header=TRUE, sep=",", na.strings=c("NA",".","","#N/A"))    #this is the test data file
  
  dat2 <- fread("morge_conthey.csv", header=TRUE, sep=";", na.strings=c("NA",".","","#N/A"))   
  complete_datetime <- seq(from = min(dat2$Time), 
                           to = max(dat2$Time), 
                           by = "5 min")
  q <- dat2$`Discharge [m3/s]`
  q[q < 0] <- 0 
  NonNAindex <- which(!is.na(q))
  meanq <- mean(q[NonNAindex])
  q[is.na(q)] <- meanq
  dat2$`Discharge [m3/s]` <- q
  # Merge with the original dataframe to fill in missing dates
  df_complete <- data.frame(Time = complete_datetime) %>%
    left_join(dat2, by = "Time") %>%
    mutate(`Discharge [m3/s]` = ifelse(is.na(`Discharge [m3/s]`), meanq, `Discharge [m3/s]`))
  
  
  date_time <- as.POSIXct(df_complete$Time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  t2 <- fractional_year(date_time)
  df_complete$t <- t2;
  
  dat <- inner_join(dat1, df_complete, by = "t")

  
  t <- dat$t
  
  p <- dat$p 
  if (is.null(p)) p <- dat$p   # to handle different column headers
  p[is.na(p)] <- 0
  
  pet <- dat$pet
  if (is.null(pet)) pet <- dat$pet  # to handle different column headers
  
  q <- dat$`Discharge [m3/s]`
  q[q < 0] <- 0 
  q[is.na(q)] <- 0
  
  # convert cubic meter per s. to mm per hour. I use that Sion has area of 80 km^2.
  q <- q*3600*1000 / (80 * 1000000);
  
  #  q <- dat$Q_cms
  # if (is.null(q)) q <- dat$Q_cms   # to handle different column headers
}
if (FALSE) {  # This data was used a sanity check to make sure the calibration tool was working correctly on simulated data.
  dat <- fread("../../Lavertezzo_ref_flow_flashy_fluxes.csv", header=TRUE, sep=",", na.strings=c("NA",".","","#N/A"))    #this is the test data file
  
  t <- dat$t
  
  p <- dat$p 
  if (is.null(p)) p <- dat$p   # to handle different column headers
  
  pet <- dat$pet
  if (is.null(pet)) pet <- dat$pet  # to handle different column headers
  
  q <- dat$q
  if (is.null(q)) q <- dat$q   # to handle different column headers
}
if (TRUE){
  dat <- fread("GISID-25_BAFU-2199.csv", header=TRUE, sep=",", na.strings=c("NA",".","","#N/A"))    #this is the test data file
  
  datetime <- dat$datetime;
  
  t <- fractional_year(datetime)
  
  p <- dat$P_mmhr 
  if (is.null(p)) p <- dat$P_mmhr   # to handle different column headers
  p[is.na(p)] <- 0
  
  pet <- dat$PET_mmhr
  if (is.null(pet)) pet <- dat$PET_mmhr  # to handle different column headers
  NonNAindex <- which(!is.na(pet))
  meanpet <- mean(pet[NonNAindex])
  pet[is.na(pet)] <- meanpet
  
  q <- dat$Q_mmhr
  if (is.null(q)) q <- dat$Q_mmhr   # to handle different column headers
  
  NonNAindex <- which(!is.na(q))
  meanq <- mean(q[NonNAindex])
  
  q[is.na(q)] <- meanq
  
  #  q <- dat$Q_cms
  # if (is.null(q)) q <- dat$Q_cms   # to handle different column headers
  
  p <- p[(24*365*4):(24*365*10)]
  q <- q[(24*365*4):(24*365*10)]
  pet <- pet[(24*365*4):(24*365*10)]
  
  }




dt <- 1

agg <- 1


n <- length(p)

# parameter values
bu <- 20.0          # upper box drainage exponent
bl <- 2.2          # lower box drainage exponent
bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
su_ref <- 500     # upper box reference storage
sl_ref <- 250    # lower box reference storage
sc_ref <- 2     # channel box storage
fw <- 0.7          # fraction of upper box storage over which ET responds to storage level
pet_mult <- 0.8   # PET multiplier
f_OF <- 0.15      # fraction of discharge in reference state from overland flow
f_SS <- 0.30       # fraction of discharge in reference state from shallow subsurface flow


# note bc is fixed at 1.5 by hydraulic geometry


# Note also that bu and su_ref will be nearly proportional to one another
# Should change optimization routine so that parameters are more nearly orthogonal



####################################
# here we fit parameters -- comment out this part if you just want to run the model with the parameter values specified above
# instead of using them as initial values for parameter fitting
####################################


params <- c(bu, bl, su_ref, sl_ref, sc_ref, fw, f_OF, f_SS)
cntrls <- list(trace=6,
               fnscale=-1,
               parscale=params, maxit = 200, factr=1e-3)

# fit parameter values by optimizing NSE on discharge (for better or worse...)
opt <- optim(par=params,
             fn=NSE, gr=NULL,
             method="L-BFGS-B",
             lower=c(1, 1, 1.0, 1.0, 1.0, 0, 0, 0),
             upper=c(50, 20, 1000.0, 2000.0, 100.0, 5, 1, 1),
             control=cntrls,
             p=p, pet=pet, q=q, dt=dt, pet_mult=pet_mult)


# copy out optimal parameter values
bu <- opt$par[1]
bl <- opt$par[2]
su_ref <- opt$par[3]
sl_ref <- opt$par[4]
sc_ref <- opt$par[5]
fw <- opt$par[6]
f_OF <- opt$par[7]
f_SS <- opt$par[8]

if (opt$convergence != 0) print( c("ERROR CODE =", opt$convergence))
print( opt$message )
print( c("opt NSE =", opt$value) )
print( c("opt bu =", bu) )
print( c("opt bl =", bl) )
print( c("fixed bc =", bc) )
print( c("opt su_ref =", su_ref) )
print( c("opt sl_ref =", sl_ref) )
print( c("opt sc_ref =", sc_ref) )
print( c("opt fw =", fw) )
print( c("opt f_OF =", f_OF) )
print( c("opt f_SS =", f_SS) )


################################
# END of parameter optimization block
# comment out the block above if you just want to run the model with the specified parameter values, without optimization
################################
