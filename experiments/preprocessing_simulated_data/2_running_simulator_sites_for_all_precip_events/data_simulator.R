
# this script allows to generate the streamflow time series


rm(list=ls())  #clear environment

# install.packages('data.table')
install.packages('dplyr')
# install.packages('Rcpp')


#library(data.table)
library(dplyr)

library(Rcpp)  #load Rcpp


sourceCpp("../code/threebox_light_v1.0_for_Rcpp.cpp")  #this is the benchmark model code (water flux module, slimmed down for parameter optimization)
sourceCpp("../code/threebox_v1.15_for_Rcpp.cpp")  #this is the benchmark model code (water flux module)


options(digits=7)
STATIONS = c('base','damped','flashy')

for (id_station in 1:length(STATIONS)){
        
    STATION = STATIONS[id_station]
    dat <-  read.csv(file.path('../../data/simulated_data/hourly_data_withPET/Lugano.csv'))
    
    dir.create(file.path(paste0('../../data/simulated_data/hourly_data_withPET/', STATION, collapse=NULL)), showWarnings = FALSE);    
    path_save = paste0('../../data/simulated_data/hourly_data_withPET/', STATION, collapse=NULL);
    path_save = paste0(path_save, '/', collapse=NULL);

    
    t <- dat$t
    p <- dat$p;
    idx_evts = which(p>1);
    idxs = which(idx_evts<length(p)-24*100);
    idx_evts = idx_evts[idxs];
    idx_evts = append(c(-1), idx_evts);
    filename = paste0(path_save, "idx_evts.rda", collapse=NULL);
    save(idx_evts, file = filename);
    
    
    for (ite in 1:length(idx_evts)){
      if (ite==1){
        p <- dat$p;
      }
      else{
        p <- dat$p;
        p[idx_evts[ite]] = 0;
      }
      if (is.null(p)) p <- dat$p   # to handle different column headers
      pet <-  dat$pet
      if (is.null(pet)) pet <- dat$PET  # to handle different column headers
      
      
      RMS <- function(x, wt=rep(1, length(x))) {  # root-mean-square average
        return(sqrt(weighted.mean(x*x, w=wt, na.rm=TRUE)))
      }
      
      weighted_se <- function(x, wt=rep(1, length(x))) {  # weighted standard error
        xbar <- weighted.mean(x, wt, na.rm=TRUE)
        std <- sqrt( weighted.mean( (x-xbar)*(x-xbar), wt, na.rm=TRUE) )
        n <- sum(!is.na(x*wt))
        return(std/sqrt(n))
      }
      
      dt <- 1
      agg <- 1
      n <- length(p)
      
    
    if ('base'==STATION){ 
      bu <- 40.      # upper box drainage exponent
      bl <- 6.          # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 100     # upper box reference storage
      sl_ref <- 1000    # lower box reference storage
      sc_ref <- 5     # channel box storage
      fw <- 1.3          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.1      # fraction of discharge in reference state from overland flow
      f_SS <- 0.6       # fraction of discharge in reference state from shallow subsurface flow
    }
    if ('damped'==STATION){ 
      bu <- 20.      # upper box drainage exponent
      bl <- 6          # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 100     # upper box reference storage
      sl_ref <- 1000    # lower box reference storage
      sc_ref <- 20     # channel box storage
      fw <- 1.10          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.50      # fraction of discharge in reference state from overland flow
      f_SS <- 0.30       # fraction of discharge in reference state from shallow subsurface flow
    }

    if ('flashy'==STATION){ 
      bu <- 50.      # upper box drainage exponent
      bl <- 2.24         # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 294.83     # upper box reference storage
      sl_ref <- 24    # lower box reference storage
      sc_ref <- 1.74     # channel box storage
      fw <- 1.06          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.13      # fraction of discharge in reference state from overland flow
      f_SS <- 0.55       # fraction of discharge in reference state from shallow subsurface flow
    }
      
      # note bc is fixed at 1.5 by hydraulic geometry
      # Note also that bu and su_ref will be nearly proportional to one another
      # Should change optimization routine so that parameters are more nearly orthogonal
      
      ##################################
      ## Benchmark hydrological model, WITHOUT parameter optimization
      
      npert <- 100        # number of perturbation time steps
      spinup <- 0         # number of time steps during spinup period
      
      
      first <- spinup+1   # for removing spinup period from results, so that further analyses are done on "first" to "last"
      last <- n
      
      
      r_et <- 1 # 0.0
      epsilon_et <- -20.0
      isotopic <-  FALSE # TRUE 
      
      sincoeff = -0.0464  
      coscoeff = -2.19 
      offset = -10
      
      ser_corr <- 0.5 # serial correlation between successive precip concentrations
      
      corr_rand <- rep(NA, n)
      corr_rand[1] <- rnorm(1)
      for (i in (2:n)) corr_rand[i] <- ser_corr*corr_rand[i-1] + sqrt(1.0-ser_corr*ser_corr) * rnorm(1);
      Cp <- sincoeff*sin(2*pi*t)  + coscoeff*cos(2*pi*t) + corr_rand * 3 + offset          # add some seasonal cycle
      Cp[(p<=0.0)] <- NA                                # null out concentration when we have no volume
      
      
      
      ############################################
      # prepare the input for Rcpp as a named list
      
      input <- list( bu = bu,  #first seven entries are model parameters
                     bl = bl,
                     bc = bc,
                     su_ref = su_ref,
                     sl_ref = sl_ref,
                     sc_ref = sc_ref,
                     fw = fw,
                     f_OF = f_OF,
                     f_SS = f_SS,
                     r_et = r_et,
                     epsilon_et = epsilon_et,
                     isotopic = isotopic,
                     nttd = 100,
                     npert = npert,       #number of perturbation time steps
                     agg = agg,
                     pet_mult = pet_mult,
                     n = n,   
                     dt = dt,
                     tol = 0.001,         # tolerance for numerical integration
                     t = t,               # time markers from input file
                     p = p,               # precip time series from input file
                     pet = pet,           # potential ET from input file
                     Cp = Cp)             # precip tracer concentration from input file
      
      
      
      
      ######################################################
      # now run the benchmark model, including perturbations
      bench <- c(input, threebox_model( input ))
            
      
      if (ite==1){
        filename = paste0(path_save, sprintf("data_%s.txt", STATION), collapse=NULL);
        df <- data.frame(t=t, p=p, pet=pet, et=bench$et, R=bench$R, q=bench$q)
        write.csv(df, filename, row.names=FALSE);
          qref <- bench$q;
      }
      else{
        filename = paste0(paste0(path_save, as.character(idx_evts[ite]), collapse=NULL), "_flow_fluxes.txt", collapse=NULL);
          tf <- qref[idx_evts[ite]:min(idx_evts[ite]+24*30,n)]-bench$q[idx_evts[ite]:min(idx_evts[ite]+24*30,n)];
          df <- data.frame(t=bench$t[idx_evts[ite]:min(idx_evts[ite]+24*30,n)], tf=tf)
   
        write.csv(df, filename, row.names=FALSE)
      }
      
    } 
    

}


