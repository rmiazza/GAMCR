#
# Scripts for ensemble rainfall-runoff analysis (ERRA):
#
#     ERRA: estimates Runoff Response Distributions (RRD's) and Nonlinear Response Functions (NRF's, broken-stick piecewise-linear estimates of nonlinear precipitation-dependence)
#             with optional splitting into subsets of precipitation time steps according to multiple filter criteria.  These are evaluated at evenly spaced lag times or, 
#             optionally, as a broken-stick piecewise-linear model evaluated over a geometric progression of lag times.
#             Correction for autoregressive/moving-average (ARMA) errors is implemented by default.  
#
# => Note that the function declaration for ERRA is at 3000.  There is an extended comment block below that function declaration, 
#             explaining the full range of options available, and explaining what the outputs mean.  
#             Users are STRONGLY encouraged to actually READ those explanations before experimenting with function calls!
#
#
#
# version 1.0  build 2024.07.09
# Author: James Kirchner, ETH Zurich
#
# Copyright (C) 2024 ETH Zurich and James Kirchner
# PUBLIC USE OF THIS SCRIPT IS NOT CURRENTLY PERMITTED.  *AFTER* the underlying paper (K2024, see below) is published, public use will be 
# permitted under GNU General Public License 3 (GPL3); for details see <https://www.gnu.org/licenses/>
# 
#
# READ THIS CAREFULLY:
# ETH Zurich and James Kirchner make ABSOLUTELY NO WARRANTIES OF ANY KIND, including NO WARRANTIES, expressed or implied, that this software is
#    free of errors or is suitable for any particular purpose.  Users are solely responsible for determining the suitability and
#    reliability of this software for their own purposes.
#
# ALSO READ THIS:
# These scripts implement ensemble rainfall runoff analysis as presented in J.W. Kirchner, Quantifying nonlinear, 
#     nonstationary, and heterogeneous hydrologic behavior using Ensemble Rainfall-Runoff Analysis (ERRA): proof of concept,
#     Hydrology and Earth System Sciences, 2024 (hereafter denoted K2024).
#     Users should cite that paper ***WHEN IT COMES OUT!***.
#
# This math underlying this approach is explained in J.W. Kirchner, Impulse response functions for heterogeneous, 
#     nonstationary, and nonlinear systems, estimated by deconvolution and demixing of noisy time series, 
#     Sensors, 22(9), 3291, https://doi.org/10.3390/s22093291, 2022 (hereafter denoted K2022).
#     Users should also cite that paper.
#
#
# Equations denoted K2019 refer to J.W. Kirchner, Quantifying new water fractions and transit time distributions using ensemble 
#     hydrograph separation: theory and benchmark tests, Hydrology and Earth System Sciences, 23, 303-349, https://doi.org/10.5194/hess-23-303-2019, 2019.
#
# Equations denoted KK2020 refer to J.W. Kirchner and J.L.A. Knapp, Technical note: Calculation scripts for ensemble hydrograph 
#     separation, Hydrology and Earth System Sciences, 24, 5539-5558, https://doi.org/10.5194/hess-24-5539-2020, 2020.
#
# The equations may differ in detail.  For example, in these scripts, lag indices range from 1 to m+1 instead of 0 to m 
# due to the array indexing conventions of the R language. 




# SOME NOTES ABOUT PERFORMANCE:
#
# These scripts call functions from IRFnnhs.R (see K2022) to do the hard work of estimating the impulse response functions linking precipitation to streamflow.
# The IRF routines must solve large matrix problems when Runoff Response Distributions (RRDs) or Nonlinear Response Functions (NRFs) are calculated over many 
# lag times from long high-frequency time series, particularly when precipitation is split into multiple subsets.  The order of difficulty for these matrix problems 
# is approximately n * (m+1)^2 * n_sets^2 * nxk^2, where n is the number of time steps, n_sets is the number of subsets that 
# precipitation is split into, nxk is the number of knots used to evaluate nonlinearities in response to precipitation, 
# and m+1 is the number of time steps over which the RRD is estimated.  
#
# Base R currently (i.e., in 2022/2023) does not use particularly efficient BLAS (Basic Linear Algebra Subprograms) routines by default.
# Switching to a processor-optimized BLAS, such as the Intel oneAPI Math Kernel Library (or MKL), in the case of computers running Intel processors, can speed up matrix operations by an 
# order of magnitude or more.  Instructions for linking the Intel MKL on Linux are at https://www.intel.com/content/www/us/en/developer/articles/technical/quick-linking-intel-mkl-blas-lapack-to-r.html
# On Windows, OpenBLAS can be installed relatively straightforwardly as described at https://github.com/david-cortes/R-openblas-in-windows.
# A web search will reveal other approaches for other operating systems and processors.
#
# Further performance gains may be possible by using the R command "options(matprod = "blas")".  This will invoke the BLAS without checking for NaN or Inf values, 
# which (if they are present), would lead to unpredictable results.  Thus this option should usually be avoided, and only be used with caution.
#
# These large matrix problems could also lead to memory constraints.  The size of the design matrix scales as n * (m+1) * n_sets * nxk.  The IRF routine saves memory 
# by creating the design matrix in discrete chunks of rows (=time steps), calculating the necessary cross-products for each chunk, and then combining the cross-products.  
# This approach exploits the fact that the m+1 columns within each precipitation subset are lags of one another, so they do not all need to be stored
# in memory.  This incurs a negligible performance penalty, and reduces the largest matrices that must be stored to order n * n_sets.
#
# For robust estimation it can be advantageous to create the design matrix all in one chunk, so that it can be re-used in each iteration rather than needing
# to be re-created.  However, if the design matrix is large enough to trigger memory paging, this advantage is lost.  The (approximate) upper limit on the size of the
# design matrix, above which it will be chunked, is determined by the parameter max.chunk.  Users who are experiencing unexpectedly slow performance can experiment
# with changing max.chunk to see whether that helps.
#
# For efficiency reasons, three time-consuming subroutines are coded in C++ under Rcpp.  



# if any of these packages are not already installed, do install.packages("<package name>")
# Windows users will also need to install rtools (see https://cran.r-project.org/bin/windows/Rtools/rtools40.html)
# for the necessary GNU C++ compiler; this should already be available for Mac and Linux implementations

library(data.table)
library(dplyr)  
library(Matrix)
library(matrixStats)  
library(Rcpp)
library(caTools)



# Global variables are in ALL CAPS


TICK <<- Sys.time()    # start stopwatch











#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
#### Here we include the functions from IRFnnhs.R, rather than sourcing IRFnnhs.R as a separate file ####
#### The version of IRFnnhs.R used here has been modified from the original, to include seg_wtd_meanx, seg_wtd_IRF, seg_wtd_ykx, and their associated standard errors ####
#### Propagation of uncertainties has also been modified to take account of potential covariances among different x's (not just different lag indices for the same x's) ####
#### Robust estimation has been modified so that the IRLS algorithm does not collapse when we have >50% identical y values (such as zero discharges in ephemeral streams) ####
#### First-differencing has also been removed, to eliminate unneeded complexity ####
#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////








#/////////////////////////////////////
#### create regularization matrix ####
#/////////////////////////////////////

tikh <- function(m) {     # creates an m x m Tikhonov - Phillips regularization matrix (equation 49 of K2019)
  
  v <- rep(0,m)           # start with a vector of zeroes
  v[1:3] <- c(6, -4, 1)    
  x <- toeplitz(v)        # make this into a toeplitz matrix
  
  #first two rows are special
  x[1, 1:2] <- c(1, -2)
  x[2, 1:3] <- c(-2, 5, -4)
  
  #last two rows are special
  x[m, (m-1):m] <- c(-2, 1)
  x[(m-1), (m-2):m] <- c(-4, 5, -2)
  
  return(x)
} # end tikh

#////////////////////////////////////
# END OF create regularization matrix
#////////////////////////////////////








#////////////////////////////////////////////////////////////////
#### Rcpp function to efficiently calculate predicted values ####
#////////////////////////////////////////////////////////////////
cppFunction('NumericVector calc_pred(      //THIS FUNCTION IS FOR USE IN SOLVAR, NOT IRF
               NumericVector x, 
               NumericVector y,
               NumericVector beta,
               int nx,
               int n,
               int nlag,
               int h) {

  int i, lag;
  
// x is an matrix of nx columns and n rows (which is handled here as if it were a vector, for efficiency and clarity)
// nlag is the number of lags in the xx matrix; h is the number of lags that we add from the y vector

  NumericVector pred(n-nlag+1, beta[beta.length()-1]);            //declare and initialize the target vector to constant term

  for (i=0; i<nx; i++) {             // step through columns of x
      for (lag=0; lag<nlag; lag++) {      // step through lags
          pred = pred + beta(i*nlag+lag)*x[Rcpp::Range((i*n+nlag-1-lag),(i*n+nlag-2-lag+pred.length()))];
      }
  }
  
  // now do lags of y for AR correction
  if (h>0) for (i=0; i<h; i++) {     // step through h
          pred = pred + beta(nx*nlag+i)*y[Rcpp::Range((nlag-1-(i+1)),(nlag-2-(i+1)+pred.length()))];
   }
  
  return pred;
  }
')





#////////////////////////////////////////////////////////////////
#### Rcpp function to efficiently calculate predicted values ####
#////////////////////////////////////////////////////////////////
cppFunction('NumericVector calc_ypred(      //THIS FUNCTION IS FOR USE IN IRF, NOT SOLVAR
               NumericVector x, 
               NumericVector beta,
               int nx,
               int n,
               int m,
               int h) {

  int i, lag, nlag;
  
  nlag = m+h+1;
  
// x is an matrix of nx columns and n rows (which is handled here as if it were a vector, for efficiency and clarity)
// nlag is the number of lags in the xx matrix; h is the number of lags that we add from the y vector

  NumericVector pred(n-nlag+1);            //declare and initialize the target vector

  for (i=0; i<nx; i++) {             // step through columns of x
      for (lag=0; lag<=m; lag++) {      // step through lags
          pred = pred + beta(i*(m+1)+lag)*x[Rcpp::Range((i*n+nlag-1-lag),(i*n+nlag-2-lag+pred.length()))];
      }
  }
  
  return pred;
  }
')










#//////////////////////////////////////////////////////
#### Rcpp function to efficiently create xx matrix ####
#//////////////////////////////////////////////////////
cppFunction('NumericVector buildxx(
               NumericVector x, 
               NumericVector y,
               int nx,
               int n,
               int nlag,
               int h,
               int chunktop,
               int chunkht) {


  NumericVector::iterator xroot = x.begin();
  NumericVector::iterator yroot = y.begin();
  NumericVector::iterator snip1;
  NumericVector::iterator snip2;
  NumericVector::iterator dest;

  int i, lag;
  
// x is an matrix of nx columns and n rows (which is handled here as if it were a vector, for efficiency and clarity)
// nlag is the number of lags in the x matrix; h is the number of lags that we add from the y vector

  NumericVector xx((nx*nlag+h+1)*chunkht);            //declare and initialize the target matrix (here shown only as a vector -- this will appear as if it were a matrix in the result)
  NumericVector::iterator xxroot = xx.begin();

  for (i=0; i<nx; i++) {             // step through columns of x
      for (lag=0; lag<nlag; lag++) {      // step through lags
          //this is the vector location where we need to start copying: move over to column i, down to chunktop (minus 1 because indexing starts at zero in C instead of 1 in R), 
          //and then back up by lag:
          snip1 = xroot + (i*n) + (chunktop-1) - lag; 
          snip2 = snip1 + chunkht;                           //this is the vector location where we should stop copying
          dest = xxroot + (i*chunkht*nlag) + lag*chunkht;    //this is where we should paste the vector in (each column of x becomes nlag columns here, each of length chunkht)
          std::copy(snip1, snip2, dest);
      }
  }
  
  // now do lags of y for AR correction (again we do this with iterators)
  if (h>0) for (i=0; i<h; i++) {     // step through h
          snip1 = yroot + (chunktop-1) - (i+1);              //step down to chunktop (with -1 due to indexing difference between the C and R languages), then back up by i+1 because i=0 corresponds to lag=1 here...
          snip2 = snip1 + chunkht;
          dest = xxroot + (nx*nlag*chunkht) + (i*chunkht);
          std::copy(snip1, snip2, dest);
  }
  
  // now add the constant column
  NumericVector unity(chunkht);
  std::fill(unity.begin(), unity.end(), 1.0);
  std::copy(unity.begin(), unity.end(), xxroot+(nx*nlag+h)*chunkht);
  return xx;
  }
')







#///////////////////////////////////////////////////////////////////////
#### calculate weighted autocorrelation and partial autocorrelation ####
#///////////////////////////////////////////////////////////////////////

wtd.pacf <-
  
  function(y,
           wt=rep(1, NROW(y)) , 
           maxlag=NULL )                    # maximum lag at which acf and pacf will be estimated
  {
    
    if (NROW(y)!=NROW(wt)) stop("error in wtd.pacf: data vector and weight vector must be same length")
    if (min(wt, na.rm=TRUE)<0) stop("error in wtd.pacf: weights cannot be less than zero")
    
    wt[is.na(wt)] <- 0                      # convert NA's to zero weight
    
    wt[is.na(y)] <- 0                       # set weight for any missing values to zero
    
    y[is.na(y)] <- 0                        # zero out NA values (but don't delete them: we need to preserve the temporal ordering!)
    
    n <- NROW(y)                            # length of input vector
    
    n.eff <- (sum(wt)^2)/sum(wt^2)          # effective sample size
    
    if (n.eff<30) stop("error in wtd.pacf: effective sample size of at least 30 is needed to estimate acf reliably")
    
    if (is.null(maxlag)) maxlag <- floor(min(n.eff/2, 10*log10(n.eff)))
    
    y <- y - sum(wt*y)/sum(wt)              # remove weighted mean from y
    
    acf <- rep(0, maxlag)                   # initialize acf vector
    pacf <- rep(0, maxlag)                  # initialize pacf vector
    
    #    y.sw <- y * sqrt(wt)                    # y weighted by square root of wt
    
    #    for (i in 1:maxlag) acf[i] <- cor(y.sw, dplyr::lag(y.sw, i), use="complete.obs")   # autocorrelation
    
    # this seems to behave more sensibly than the approach above with robustness weights (which makes sense because the full robustness weight is applied at each time step, not the square root) 
    for (i in 1:maxlag) acf[i] <- cor(y*wt, dplyr::lag(y, i), use="complete.obs")   # autocorrelation 
    
    vacf <- c(1, acf[-length(acf)])         # copy of autocorrelation, now for lags of zero to maxlag-1  (for toeplitz matrix that we use to calculate pacf)
    
    pacf[1] <- acf[1]/vacf[1]               # first-order partial autocorrelation equals autocorrelation at lag 1, divided by autocorrelation at lag zero (which =1)
    
    for (i in 2:maxlag) pacf[i] <- solve(toeplitz(vacf[1:i]), acf[1:i])[i]  # matrix solution for pacf
    # see, for example, https://stats.stackexchange.com/questions/129052/acf-and-pacf-formula
    
    return(list(
      acf=acf, 
      pacf=pacf))
    
  }










#////////////////////////////////////////////////////////////////////////////
#### calculate cross-products, (with chunking of xx matrix if necessary) ####
#////////////////////////////////////////////////////////////////////////////

calc_crossprods <-
  function(y,                               # any missing values must have been converted to zero
           x,                               # any missing values must have been converted to zero!
           wt=rep(1, NROW(x)) ,             # this must already have zeroes for any rows where x or its lags will be NA
           maxlag ,                         # this is n_lags in calling routine; does not include 1 more for lag zero
           h ,
           xx = NULL ,
           chunk.maxht ,
           verbose = FALSE )
  {
    
    nx <- NCOL(x)
    n <- NROW(x)
    nlags <- maxlag+1                                          # add one to account for lag zero
    wt[1:maxlag] <- 0                                          # just in case this hasn't been done already... the first maxlag rows will not be valid b/c one or more lags are missing
    sqwt <- as.vector(sqrt(wt))                                # sw is square root of weight (including robustness weights)
    
    if (!is.null(xx)) {                                        
      
      # if we already have a one-piece xx matrix, we can use it directly without compiling it again
      C <- crossprod(sqwt[(maxlag+1):n]*xx)                                        # use R's very fast cross-product routine (fast with a good BLAS, that is)
      xy <- crossprod(sqwt[(maxlag+1):n]*xx, sqwt[(maxlag+1):n]*y[(maxlag+1):n])   # note that because x and y are multiplied by sqrt(wt), these cross-products are weighted
      n.nz <- colSums(((xx*sqwt[(maxlag+1):n]) != 0))                              # count non-zero elements (with non-zero weight) in each x column
      
    } else {                                                   
      
      # if we don't already have an xx matrix, we'll have to build it, potentially with chunking
      # initialize outputs at zero because we'll be adding to them with each chunk
      C <- matrix(0, nrow=nx*nlags + h + 1, ncol=nx*nlags + h + 1)     # cross product matrix has an extra row and column for the constant term
      xy <- rep(0, nx*nlags + h + 1)                                   # x-y cross products (with an extra row for the constant term)
      n.nz <- rep(0, nx*nlags + h + 1)                                 # this vector tallies non-zero elements of x (with nonzero weight)
      
      
      chunkend <- maxlag                                           # initializing this at the last row that's guaranteed to be infeasible due to NA's in lags;
      # it'll change as soon as we enter the loop to build the xx matrix
      
      repeat{
        chunktop <- chunkend + 1                                   # new chunk starts on next step after the last one ended
        chunkend <- min(chunktop+chunk.maxht-1, n)                 # new chunk ends at chunk.top+chunk.maxht (or at the end of the data set)
        chunkht <- chunkend-chunktop + 1                           # height of this chunk
        
        if (sum(sqwt[chunktop:chunkend])>0) {                      # if all of the rows in this chunk have weight of zero then we can skip this one
          xx <- buildxx(x, y, nx, n, nlags, h, chunktop, chunkht)  # here is the Rcpp routine to efficiently build the xx matrix
          xx <- matrix(xx, nrow=chunkht)
          yy <- y[chunktop:chunkend]                               # take corresponding chunk of y
          sw <- sqwt[chunktop:chunkend]                            # and corresponding chunk of sqrt(weight)
          
          C <- C + crossprod(sw*xx)                                # use R's very fast cross-product routine (fast with a good BLAS, that is)
          xy <- xy + crossprod(sw*xx, sw*yy)                       # note that because x and yy are multiplied by sqrt(wt), these cross-products are weighted
          
          n.nz <- n.nz + colSums(((xx*sw) != 0))                   # count non-zero elements (with non-zero weight) in each x column and add to any tallies from previous chunks
          
        } # end if
        
        if (verbose) if (chunkend<n) cat(paste0(ceiling(100*chunkend/n), "% of cross-products done...: ", round(difftime(Sys.time(), TICK, units="secs"), 3), "                           \r"))
        
        if (chunkend >= n) {break}                                 # exit if this is the last chunk
      } # end repeat
      
    } # end else (i.e., we don't already have an xx matrix)
    
    
    if (chunk.maxht>=n) {
      return(list(                                                 # if we have a complete xx matrix, pass it along so it can be re-used...
        C = C ,
        xy = xy ,
        n.nz = n.nz ,
        xx = xx))
    } else {
      return(list(
        C = C ,
        xy = xy ,
        n.nz = n.nz ,
        xx = NULL))
    }
  }



















#////////////////////////////////////////////////////////////////////////////////////////////
#### Robust solution to matrix equations for IRF, with chunking of x matrix if necessary ####
#////////////////////////////////////////////////////////////////////////////////////////////

RobustSolvAR <-
  function(y, 
           x, 
           wt ,
           h = 0 ,
           m ,
           nu = 0 ,
           robust = FALSE ,
           verbose = FALSE,
           chunk.maxht = chunk.maxht)
    
  {
    
    # takes as input:
    # y              vector of y values
    # x              matrix of x values
    # wt             vector of weight values
    # h              order of optional AR correction
    # m              integer value of maximum lag (first lag is zero, so total number of lags is m+1).  This doesn't count additional lags needed for AR correction or first differences  
    # nu             fractional weight, 0-1, to be given to Tikhonov-Phillips regularization (default is 0 = no regularization)
    # robust         flag controlling whether robust estimation by Iteratively Reweighted Least Squares (IRLS) will be used
    # verbose        controls whether progress reports are printed (TRUE) or suppressed (FALSE)
    # chunk.maxht    maximum height of design matrix that will not trigger chunking
    
    # returns a list as output, with the following objects:
    # b.hat          regression coefficients (vector of length nx*(m+1))
    # se             standard errors (vector of length nx*(m+1))
    # Kbb            covariance matrix of coefficients, with nx*(m+1) rows and columns
    # phi            fitted AR coefficients (vector of length=h)
    # n              total number of time steps
    # n.eff          effective number of time steps, accounting for uneven weighting
    # n.nz           number of nonzero elements in the *weighted* x's (so values with zero weight don't count); vector of length nx*(m+1)
    # e              residuals
    # s2e            weighted residual variance
    # resid.acf      autocorrelation function of residuals
    # resid.pacf     partial autocorrelation function of residuals
    # rw             vector of robustness weights
    
    
    # If robust==TRUE, runtimes may be substantially faster if the robust solvAR can avoid compiling the cross-products in chunks, and instead create the
    # entire xx matrix all at once, so it can be re-used for each iteration of the robust estimation loop.  
    # This requires enough memory to hold this matrix (and potentially several copies of it) without triggering memory paging. 
    # On Windows, use memory.limit(size=###) to change the memory available to R, where ### is in MB.  
    
    
    maxlag <- m+h                   # total number of lags (not counting one for lag zero)
    
    nx <- NCOL(x)                   # this is the number of different x variables
    
    mm <- (maxlag+1)*nx             # total number of b coefficients (and width of the x matrix, not counting the AR terms or the constant term)
    
    iter_limit <- 200               # iteration limit for robust estimation
    
    
    
    #///////////////////////////////////
    # zero out missing values of y and x
    
    orig_y <- y                                               # first make copies of y and x (for calculating residuals later)
    orig_x <- x
    orig_wt <- wt
    
    # note that all of these cases, and their lags, correspond to rows where wt has already been zeroed (in IRF), 
    # so replacing them with zeroes will not affect the values of the cross products (but will keep crossprod from choking on NA's)
    y[is.na(y)] <- 0                                          # replace missing values of y with zeroes
    x[is.na(x)] <- 0                                          # replace missing values of x with zeroes
    
    n <- length(y)                                            # number of rows
    
    if (nu>0) {                                               # if we regularize...:
      # We can't use one big regularization matrix for the whole problem, because it will lead to leakage between the end of one set of b coefficients and the start of another. 
      # Instead we need to create a segmented regularization matrix, with nx Tikhonov-Phillips matrices along the diagonal
      tt <- tikh(1+maxlag)                                    # start with one Tikhonov-Phillips matrix (this is our building block)
      H <- matrix(0, nrow = mm+h+1, ncol=mm+h+1)              # start with a matrix of all zeroes
      for (j in 1:nx)                                         # copy our building block into the right range(s) along the diagonal of H
        H[((j-1)*(1+maxlag)+1):(j*(1+maxlag)) , ((j-1)*(1+maxlag)+1):(j*(1+maxlag)) ] <- tt
    }    
    
    
    rw <- rep(1, n)                                           # initial values of robustness weights are all 1  NOTE: in contrast to earlier versions, here rw is of length n rather than n-maxlag.  That makes everything cleaner
    old_rw <- rw
    
    #////////////////////////////
    # robustness loop starts here
    iter <- 0                                                 # iteration counter
    xx <- NULL                                                # this will force evaluation of xx
    
    repeat{
      
      iter <- iter+1                                          # update counter
      
      
      wt <- rw*orig_wt                                        # multiply weights by robustness weight
      shortwt <- wt[(maxlag+1):n]                             # to avoid subsetting wt multiple times below
      
      sumwt <- sum(wt, na.rm=TRUE)
      
      
      # here is where we construct the xx matrix (with chunking if necessary) and calculate the crossproduct matrix C and the crossproduct vector xy
      # list2env will create local values of C, xy, n.nz, and xx.  xx will be NULL if chunking was required.
      list2env(    calc_crossprods(y=y, x=x, wt=wt, maxlag=maxlag, h=h, xx=xx, chunk.maxht=chunk.maxht, verbose=verbose),    envir=environment())
      
      C <- C/sumwt                                           # dividing by the sums of the weights does not change the regression parameters, 
      xy <- xy/sumwt                                         # but is necessary for consistency with the way uncertainties are calculated next
      
      if (nu==0) b.hat <- solve(C, xy)                       # if we *don't* regularize, solve for the un-regularized regression coefficients
      else {                                                 # if we *do* regularize...:
        # convert weighting parameter nu (values 0 - 1) to lambda as used in conventional Tikhonov-Phillips regularization formulas
        lambda <- nu / (1.0 - nu) * sum(diag(C)[1:(nx*(1+maxlag))]) / sum(diag(H))              # equation 50 of K2019 (omitting the diagonal elements that correspond to the constant term and AR terms)
        b.hat <- solve((C + (lambda * H)), xy) # now solve (regularized) linear system          # equation 46 of K2019, and equation A12 of KK2020
      }  
      
      # calculate predicted values -- note that rows with NA's will not count anyhow because they will have weight of zero in calculating error variance
      # note that our predictions don't include the first maxlag rows (because our xx matrix doesn't either...)
      
      if (!is.null(xx)) {                                     # if we have the xx matrix, we can calculate the predicted y values by matrix multiplication methods (fast with good BLAS)
        pred <- xx %*% b.hat
      } else {
        pred <- calc_pred(y=y, x=x, beta=b.hat, nx=nx, n=n, nlag=maxlag+1, h=h)
        pred[shortwt==0] <- NA
      }
      
      oy <- orig_y[(maxlag+1):n]
      e <- oy - pred                                          # these are the residuals  (equation A13 of KK2020)
      e[shortwt==0] <- NA                                     # exclude all residuals for excluded rows (rows with zero weight)
      
      sumwt <- sum(shortwt)
      n.eff <- sumwt^2/sum(shortwt*shortwt)                   # calculate the effective sample size (accounting for uneven weighting)
      
      e2 <- e*e                                               # squared errors
      
      s2e <- (n.eff-1)/(n.eff-(mm+h+1)-1) * sum( e2 * shortwt, na.rm=TRUE)/sumwt   # weighted residual variance, corrected for degrees of freedom (equation ####)
      
      if (robust==FALSE) {break}                              # if we're not doing robust estimation, we can jump out of the loop here
      
      # now do iteratively reweighted least squares with some special tweaks to avoid collapse when more than the half of y's are identical
      idents <- max(table(oy[!is.na(e)], useNA="no"))                                                       # count of most common identical value in y's (excluding cases with no residuals) 
      med.e <- sqrt(quantile(e2, na.rm=TRUE, probs=(0.5+0.5*idents/sum(!is.na(e2))) ))                # median absolute residual, but with quantile adjusted upward from 0.5 to account for identical y's so that algorithm doesn't collapse with (say) > 50% y=0 values
      rw <- ifelse(is.na(e), 1, 1/(1+(e/(3.536*med.e))^2))                                                  # Cauchy weight function
      
      rw <- c(rep(0, times=maxlag), rw)                       # add leading zeroes to bring rw to length n
      
      
      # exit criterion: 99 percent of the robustness weights have changed by less than 0.1 (absolute value) since previous iteration
      if (quantile(abs(rw-old_rw)[shortwt!=0], 0.99)<0.1) {  # don't count rw's and residuals for rows that have no weight!!
        if (verbose) cat("robustness iteration ", iter, " complete at time ", round(difftime(Sys.time(), TICK, units="secs"), 3),
                         "  s2e=", round(s2e, 8), " max e=", round(sqrt(max(e2, na.rm=TRUE)), 3),
                         "rw chg= ", round(quantile(abs(rw-old_rw)[shortwt!=0], 0.99), 3), "\n")
        {break}                            
      }
      
      if (verbose) cat("robustness iteration ", iter, " complete at time ", round(difftime(Sys.time(), TICK, units="secs"), 3),
                       "  s2e=", round(s2e, 8), " max e=", round(sqrt(max(e2, na.rm=TRUE)), 3),
                       "rw chg= ", round(quantile(abs(rw-old_rw)[shortwt!=0], 0.99), 3), "\n")
      
      old_rw <- rw            # save a copy of robustness weights to compare the next iteration with
      
      if (iter>iter_limit) stop("Error: iteration limit exceeded in robust estimation loop of RobustSolvAR")
      
    } # end robustness loop
    
    
    pacf <- wtd.pacf(e, shortwt)                     # we need to calculate *weighted* partial autocorrelations... using homebaked function above                             
    resid.pacf <- pacf$pacf
    resid.acf <- pacf$acf
    
    
    # Now calculate the parameter covariance matrix. Note that scaling needs to be correct here.  
    # If C is a covariance matrix (which it is), then s2e needs to be the variance of residuals (which it is).
    
    # Because we only want the uncertainties in the b's, treating the phi's as constant (which benchmark tests 
    # show is the right thing to do -- otherwise the standard errors are inflated), we need to truncate the covariance matrix C to exclude 
    # the rows and columns corresponding to the phi coefficients.
    
    if (h==0) clipC <- C
    else clipC <- C[-(mm+1):-(mm+h) , -(mm+1):-(mm+h)]                       # remove rows and columns corresponding to phi terms
    
    if (nu==0.0) Kbb = (s2e / n.eff) * solve(clipC)                          # covariance matrix of coefficients without regularization (equation ####)
    else {
      if (h==0) clipH <- H
      else clipH <- H[-(mm+1):-(mm+h) , -(mm+1):-(mm+h)]                     # remove rows and columns corresponding to phi terms
      regC.inverse = solve((clipC + (lambda * clipH)))                       # inverse of regularized C matrix
      Kbb = (s2e / n.eff) * regC.inverse %*% clipC %*% regC.inverse          # covariance matrix of regularized coefficients (equation A15 of KK2020)
    }                                                                        # note that Kbb now has dimensions of mm+1 by mm+1 (i.e., including the constant term)
    
    
    
    # now shape output
    
    if (h==0) {    
      phi <- NA                                                              # if there is no AR correction, save phi as NA
    } else {                                                                 # if there is an AR correction...:
      phi <- b.hat[(length(b.hat)-h):(length(b.hat)-1)]                      # extract phi from fitted coefficients
    }
    
    keep.mask <- as.logical(rep(c(rep(1, (m+1)), rep(0, h)), nx))            # mask for which coefficients to keep (1's) and drop (0's)
    
    b.hat <- b.hat[1:mm][keep.mask]                                          # vector of b values (nx*(m+1) values covering all x's and lags; constant has been dropped)
    Kbb <- Kbb[1:mm, 1:mm][keep.mask, keep.mask]                             # parameter covariance matrix (dimension nx*(m+1) by nx*(m+1))
    se <- sqrt(diag(Kbb))                                                    # standard errors: diagonal of covariance matrix
    n.nz <- n.nz[1:mm][keep.mask]                                            # vector of n.nz values (nx*(m+1) values covering all x's and lags; constant has been dropped)
    
    
    b.hat <- matrix(b.hat, ncol=nx)                                          # wrap the b vector into a matrix (with one column for each x variable)
    se <- matrix(se, ncol=nx)                                                # wrap seinto a matrix with one column for each x
    n.nz <- matrix(n.nz, ncol=nx)                                            # wrap n.nz (from our original problem, not after AR correction) into a matrix
    
    
    
    
    return(
      list(
        b.hat = b.hat ,    # regression coefficients (matrix of nx columns and m+1 rows)
        se = se ,           # standard errors (matrix of nx columns and m+1 rows)
        Kbb = Kbb ,          # parameter covariance matrix (nx*(m+1) by nx*(m+1))
        n = n ,               # total number of time steps
        phi = phi ,            # fitted AR coefficients
        n.eff = n.eff ,         # effective number of time steps, accounting for uneven weighting
        n.nz = n.nz ,            # number of nonzero x values, with nonzero weight, in each column of design matrix (vector of length nx*(m+1))
        e = e ,                   # residuals 
        s2e = s2e ,                # weighted residual variance
        resid.acf = resid.acf ,     # autocorrelation function of residuals
        resid.pacf = resid.pacf ,    # partial autocorrelation function of residuals
        rw = rw                       # robustness weights (vector of length n)
      )
    )
    
    
  } # end RobustSolvAR


#/////////////////////////////////////////////
# END OF robust solution to matrix equations #
#/////////////////////////////////////////////















#/////////////////////////////////////////////////////////////////
#### robust solution to matrix equations for broken stick IRF ####
#/////////////////////////////////////////////////////////////////
BrokenStickRobustSolvAR <-
  function(y, 
           x, 
           wt ,
           knots = c(0, 1, 2, 4, 8, 16, 32, 64) ,
           h = 0 ,
           nu = 0 ,
           robust = FALSE ,
           verbose = FALSE )
    
  {
    
    
    # takes as input:
    # y              vector of y values
    # x              matrix of x values
    # wt             vector of weight values
    # knots          vector of lags at knots for which piecewise linear IRF will be evaluated (in integer number of time steps)
    #                          must be positive integers in ascending order, with first value of zero (these are checked)
    # h              order of optional AR correction
    # nu             fractional weight, 0-1, to be given to Tikhonov-Phillips regularization (default is 0 = no regularization)
    # robust         flag controlling whether robust estimation by Iteratively Reweighted Least Squares (IRLS) will be used
    # verbose        controls whether progress reports are printed (TRUE) or suppressed (FALSE)
    
    # returns a list as output, with the following objects:
    # b.hat          regression coefficients (matrix of nx columns and nk rows)
    # se             standard errors (matrix of nx columns and nk rows)
    # Kbb            covariance matrix of regression coefficients (matrix of dimensions nx*nk by nx*nk)
    # allb           interpolated regression coefficients for each lag between 0 and m (matrix of nx columns and knot(nk)+1 rows)
    # n              total number of time steps
    # phi            fitted AR coefficients
    # n.eff          effective number of time steps, accounting for uneven weighting
    # n.nz           number of nonzero elements in the *weighted* x's (so values with zero weight don't count), stored as vector of length nx*nk
    # e              residuals
    # s2e            weighted residual variance
    # resid.acf      autocorrelation function of residuals
    # resid.pacf     partial autocorrelation function of residuals
    # rw             vector of robustness weights
    
    
    # Note that in order to keep the code relatively simple, the broken-stick solvAR does not compile the cross-products in chunks, but instead creates the
    # entire xx matrix all at once.  Thus it requires enough memory (either built-in or virtual) for this matrix (and potentially several copies of it).
    # On Windows, use memory.limit(size=###) to change the memory available to R, where ### is in MB. 
    
    
    iter_limit <- 200               # iteration limit for robust estimation
    
    
    if (min(knots) < 0) stop("Fatal error in BrokenStickRobustSolvAR: knots cannot be negative")
    if (any(knots != trunc(knots))) stop("Fatal error in BrokenStickRobustSolvAR: knots must be integer-valued")
    if (min(diff(knots)) <= 0) stop("Fatal error in BrokenStickRobustSolvAR: knots must be in ascending order, with no duplicates")
    if (knots[1] != 0) stop("Fatal error in BrokenStickRobustSolvAR: first knot must be zero")
    
    nk <- length(knots)             # number of knots
    
    maxlag <- knots[nk]+h          # maximum lag (not counting one for lag zero)
    
    
    #///////////////////////////////////////////////////
    # build list of filter vectors for each of the knots
    
    # we will do this by brute force to make the underlying procedure explicit  (from Equation 52 of K2022)
    tauwt <- list()                                        # declare a list
    tauwt[[1]] <- 1                                        # start with the identity filter for lag zero
    for (k in 2:(nk-1)) {                                  # step through the other knots
      ww <- rep(0, knots[k+1])                             # note that because lags begin at zero (index=1 means lag=0) this vector will end at one lag *before* the next knot (which is OK)
      # ramp up from knots[k-1] to (and including) knots[k]
      for (i in knots[k-1]:knots[k]) ww[i+1] <- (i - knots[k-1])/(knots[k] - knots[k-1]) 
      
      # ramp down from knots[k] to knots[k+1] (not including either end, because knots[k] has already been done, and knots[k+1] would have a value of zero anyhow)
      if (knots[k+1]-knots[k] > 1) for (i in (knots[k]+1):(knots[k+1]-1)) ww[i+1] <- (knots[k+1] - i)/(knots[k+1] - knots[k]) 
      tauwt[[k]] <- ww                                     # add to list
    }
    
    ww <- rep(0, knots[nk]+1)                              # last filter vector
    for (i in knots[nk-1]:knots[nk]) ww[i+1] <- (i - knots[nk-1])/(knots[nk] - knots[nk-1]) 
    tauwt[[nk]] <- ww                                      # and add it to the list
    
    # end of filter vector construction
    
    
    nx <- NCOL(x)                   # this is the number of different x variables
    x <- as.matrix(x)               # convert x to a matrix if it isn't one
    
    n.terms <- nk+h                 # n.terms is the number of coefficients we need for each of the nx.
    
    mm <- n.terms*nx                # number of columns we will need in xx (plus h lags of y, plus 1 for the constant term)
    
    
    
    #///////////////////////////////////
    # zero out missing values of y and x
    
    orig_y <- y                                               # first make copies of y and x (for calculating residuals later)
    orig_x <- x
    orig_wt <- wt
    
    # note that all of these cases, and their lags, correspond to rows where wt has already been zeroed (in IRF), 
    # so replacing them with zeroes will not affect the values of the cross products (but will keep crossprod from choking on NA's)
    y[is.na(y)] <- 0                                          # replace missing values of y with zeroes
    x[is.na(x)] <- 0                                          # replace missing values of x with zeroes
    
    
    n <- length(y)                                            # number of rows
    
    xx <- matrix(0, nrow=n, ncol=mm+h+1)                      # design matrix with filtered time series for each knot and each x, plus AR terms and the constant term
    
    
    for (k in 1:nk) {
      xf <- stats::filter(x, filter=tauwt[[k]], method="c", sides=1)  # now apply convolutional filter for each knot to x's (xf is filtered x) -- note that doing this by FFT is MUCH slower!!
      for (i in 1:nx) xx[ , ((i-1)*n.terms+k)] <- xf[ , i]            # and distribute columns to the correct columns of xx
    }
    
    
    if (h>0) for (i in 1:nx) for (j in 1:h) xx[ , ((i-1)*n.terms+nk+j)] <- dplyr::lag(x[,i], knots[nk]+j)  # add one additional lag of each x for each AR term
    
    if (h>0) for (j in 1:h) xx[ , (mm+j)] <- dplyr::lag(y,j)     # copy AR terms, if any, into design matrix
    
    xx[ , ncol(xx)] <- 1                                       # don't forget the constant term...
    
    xx[1:maxlag, ] <- 0                                    # zero out any rows that will have NAs in filtered columns (these rows will all have zero weight anyhow)
    
    if (nu>0) {                                            # if we regularize...:
      # We can't use one big regularization matrix for the whole problem, because it will lead to leakage between the end of one set of b coefficients and the start of another. 
      # Instead we need to create a segmented regularization matrix, with nx Tikhonov-Phillips matrices along the diagonal
      tt <- tikh(n.terms)                                  # start with one Tikhonov-Phillips matrix (this is our building block)
      H <- matrix(0, nrow = mm+h+1, ncol=mm+h+1)           # start with a matrix of all zeroes
      for (j in 1:nx)                                      # copy our building block into the right range(s) along the diagonal of H
        H[((j-1)*n.terms+1):(j*n.terms) , ((j-1)*n.terms+1):(j*n.terms) ] <- tt
    }
    
    
    
    rw <- rep(1, n)                                          # robustness weights are initialized to 1
    old_rw <- rw                                             # save old robustness weights for comparison later
    
    #////////////////////////////
    # robustness loop starts here
    iter <- 0                                                # iteration counter
    repeat{
      
      iter <- iter+1                                         # update counter
      
      wt <- rw*orig_wt                                       # multiply weights by robustness weight
      sumwt <- sum(wt, na.rm=TRUE)
      sw <- as.vector(sqrt(wt))                              # sw is square root of weight (including robustness weights)
      
      C <- crossprod(xx*sw)                                  # use R's very fast cross-product routine (fast with a good BLAS, that is)
      xy <- crossprod(xx*sw, y*sw)                           # note that because x and yy are multiplied by sqrt(wt), these cross-products are weighted
      
      C <- C/sumwt                                           # dividing by the sums of the weights does not change the regression parameters, 
      xy <- xy/sumwt                                         # but is necessary for consistency with the way uncertainties are calculated next
      
      if (nu==0) b.hat <- solve(C, xy)                       # if we *don't* regularize, solve for the un-regularized regression coefficients
      else {                                                 # if we *do* regularize...:
        # convert weighting parameter nu (values 0 - 1) to lambda as used in conventional Tikhonov-Phillips regularization formulas
        lambda <- nu / (1.0 - nu) * sum(diag(C)[1:(nx*(1+n_lags))]) / sum(diag(H))                  # equation 50 of K2019 (omitting the diagonal elements that correspond to the constant term and AR terms)
        b.hat <- solve((C + (lambda * H)), xy) # now solve (regularized) linear system              # equation 46 of K2019, and equation A12 of KK2020
      }  
      
      # calculate predicted values -- note that rows with NA's will not count anyhow because they will have weight of zero in calculating error variance
      # need to use original x and y (which include NAs)
      pred <- rep(tail(b.hat,1), n)                          # initialize predicted values with the constant (the last element of b.hat)
      
      # now we need to create a matrix of beta values for each lag, linearly interpolated among the nk knots, for each of the x variables
      allb <- matrix(NA, nrow=knots[nk]+1, ncol=nx)
      
      for (j in 1:nx) {   # now step through each of the x's
        
        bk <- b.hat[((j-1)*n.terms+1):((j-1)*n.terms+nk), 1]     # subset the knot values of b for this x variable
        
        for (k in 2:nk) { 
          for (i in knots[k-1]:knots[k]) allb[(i+1), j] <- bk[k-1]*(knots[k]-i)/(knots[k]-knots[k-1]) + bk[k]*(i-knots[k-1])/(knots[k]-knots[k-1])  # linear interpolation
        }
        
        pred <- pred + stats::filter(x[, j], filter=allb[ ,j], method="c", sides=1)  # use these interpolated b's to predict y (and keep them for later, for AR correction of beta...)
      } # next j
      
      if (h>0) for (j in 1:h) pred <- pred + dplyr::lag(y,j)*b.hat[mm+j] # include AR terms in predicted values


      e <- orig_y - pred                                                       # these are the residuals  (equation A13 of KK2020)
      
      n.nzwt <- sum((wt != 0), na.rm=TRUE)                                     # effective length of the x-matrix
      sumwt <- sum(wt)

      e2 <- e*e                                                                # squared errors
      
      s2e <- (n.nzwt-1)/(n.nzwt-(mm+h+1)-1) * sum( e2 * wt, na.rm=TRUE)/sumwt  # weighted residual variance, corrected for degrees of freedom (equation ####)
      
      if (robust==FALSE) {break}                                               # if we're not doing robust estimation, we can jump out of the loop here
      
      # now do iteratively reweighted least squares with some special tweaks to avoid collapse when more than the half of y's are identical
      idents <- max(table(orig_y[!is.na(e)], useNA="no"))                                             # count of most common identical value in y's (excluding cases with no residuals) 
      med.e <- sqrt(quantile(e2, na.rm=TRUE, probs=(0.5+0.5*idents/sum(!is.na(e2))) ))                # median absolute residual, but with quantile adjusted upward from 0.5 to account for identical y's so that algorithm doesn't collapse with (say) > 50% y=0 values
      rw <- ifelse(is.na(e), 1, 1/(1+(e/(3.536*med.e))^2))                                            # Cauchy weight function
      
      
      # exit criterion: 99 percent of the robustness weights have changed by less than 0.1 (absolute value) since previous iteration
      if (quantile(abs(rw-old_rw)[wt!=0], 0.99)<0.1) {  # don't count rw's and residuals for rows that have no weight!!
        if (verbose) cat("robustness iteration ", iter, " complete at time ", round(difftime(Sys.time(), TICK, units="secs"), 3),
                         "  s2e=", round(s2e, 8), " max e=", round(sqrt(max(e2, na.rm=TRUE)), 3),
                         "rw chg= ", round(quantile(abs(rw-old_rw)[wt!=0], 0.99), 3), "\n")
        {break}                            
      }
      
      if (verbose) cat("robustness iteration ", iter, " complete at time ", round(difftime(Sys.time(), TICK, units="secs"), 3),
                       "  s2e=", round(s2e, 8), " max e=", round(sqrt(max(e2, na.rm=TRUE)), 3),
                       "rw chg= ", round(quantile(abs(rw-old_rw)[wt[(maxlag+1):n]!=0], 0.99), 3), "\n")
      
      old_rw <- rw
      
      if (iter>iter_limit) stop("Error: iteration limit exceeded in robust estimation loop of BrokenStickRobustSolvAR")
      
    } # end robustness loop
    
    
    
    resid.pacf <- as.vector(pacf(e, plot=FALSE, na.action=na.pass)$acf)      # residual ACF and PACF
    resid.acf <- as.vector(acf(e, plot=FALSE, na.action=na.pass)$acf)
    
    n.nzwt <- sum((wt[!is.na(e)] != 0), na.rm=TRUE)                          # effective length of the x-matrix (even though we've avoided creating one here...)
    sumwt <- sum(wt[!is.na(e)], na.rm=TRUE)
    
    s2e <- (n.nzwt-1)/(n.nzwt-(mm+h+1)-1) * sum(e * e * wt, na.rm=TRUE)/sumwt   # weighted residual variance, corrected for degrees of freedom (equation ####)
    
    n.eff <- sumwt^2/sum(wt*wt)                                              # calculate the effective sample size (accounting for uneven weighting)
    
    
    # Now calculate the parameter covariance matrix. Note that scaling needs to be correct here.  
    # If C is a covariance matrix (which it is), then s2e needs to be the variance of residuals (which it is).
    
    # Because we only want the uncertainties in the b's, treating the phi's as constant (which benchmark tests 
    # show is the right thing to do -- otherwise the standard errors are inflated), we need to truncate the covariance matrix C to exclude 
    # the rows and columns corresponding to the phi coefficients.
    
    if (h==0) clipC <- C
    else clipC <- C[-(mm+1):-(mm+h) , -(mm+1):-(mm+h)]                       # remove rows and columns corresponding to phi terms
    
    if (nu==0.0) Kbb = (s2e / n.eff) * solve(clipC)                          # covariance matrix of coefficients without regularization (equation ####)
    else {
      if (h==0) clipH <- H
      else clipH <- H[-(mm+1):-(mm+h) , -(mm+1):-(mm+h)]                     # remove rows and columns corresponding to phi terms
      regC.inverse = solve((clipC + (lambda * clipH)))                       # inverse of regularized C matrix
      Kbb = (s2e / n.eff) * regC.inverse %*% clipC %*% regC.inverse          # covariance matrix of regularized coefficients (equation A15 of KK2020)
    }
    
    
    
    # now shape output
    
    if (h==0) {
      phi <- NA                                                              # if there is no AR correction, save phi as NA
    } else {                                                                 # if there is an AR correction...:
      phi <- b.hat[(length(b.hat)-h):(length(b.hat)-1)]                      # extract phi from fitted coefficients
    }
    
    n.nz <- colSums(((xx*wt) != 0))                                          # count non-zero elements (with non-zero weight) in each x column
    
    
    keep.mask <- as.logical(rep(c(rep(1, nk), rep(0, h)), nx))               # mask for which coefficients to keep (1's) and drop (0's)
    
    b.hat <- b.hat[1:mm][keep.mask]                                          # vector of b values (nx*nk values covering all x's and knots; constant has been dropped)
    Kbb <- Kbb[1:mm, 1:mm][keep.mask, keep.mask]                             # parameter covariance matrix (dimension nx*nk by nx*nk
    se <- sqrt(diag(Kbb))                                                    # standard errors: diagonal of covariance matrix
    n.nz <- n.nz[1:mm][keep.mask]                                            # vector of n.nz values (nx*nk values covering all x's and knots; constant has been dropped)

    
    b.hat <- matrix(b.hat, ncol=nx)                                          # wrap the b vector into a matrix (with one column for each x variable)
    se <- matrix(se, ncol=nx)                                                # wrap se into a matrix with one column for each x
    n.nz <- matrix(n.nz, ncol=nx)                                      # wrap n.nz (from our original problem, not after AR correction) into a matrix
    
    
    
    return(
      list(
        b.hat = b.hat ,   # regression coefficients (matrix of nx columns and nk rows)
        se = se ,          # standard errors (matrix of nx columns and nk rows)
        Kbb = Kbb ,         # covariance matrix of regression coefficients (dimension nx*nk by nx*nk)
        allb = allb ,        # matrix of interpolated regression coefficients for each lag between 0 and m (nx columns and m+1 rows)
        n = n ,               # total number of time steps
        phi = phi ,            # fitted AR coefficients
        n.eff = n.eff ,         # effective number of time steps, accounting for uneven weighting
        n.nz = n.nz ,            # number of nonzero x values, with nonzero weight, in each column of design matrix (matrix of nx columns and nk rows)
        e = e ,                   # residuals 
        s2e = s2e ,                # weighted residual variance
        resid.acf = resid.acf ,     # autocorrelation function of residuals
        resid.pacf = resid.pacf ,    # partial autocorrelation function of residuals
        rw = rw                       # vector of robustness weights
      )
    )
    
    
  } # end BrokenStickRobustSolvAR

#//////////////////////////////////////////////////////////
# END OF robust solution to broken stick matrix equations #
#//////////////////////////////////////////////////////////
















#////////////////////////////////////////////////////////////////////////////////////////////////
#### IRF - Impulse Response Function estimates by least squares with correction for AR noise ####
#////////////////////////////////////////////////////////////////////////////////////////////////


IRF <-
  function(y ,
           x ,
           xnames = NULL ,
           wt = rep(1, NROW(x)) ,
           m = 60 ,
           nk = 0 ,
           nk_buffer = TRUE ,
           nu = 0 ,
           h = NULL ,
           ARprob = 0.05 ,
           ARlim = 0.2 ,
           max.AR = 12 ,
           complete = FALSE ,
           verbose = FALSE ,
           robust = FALSE ,
           max.chunk = 2e8 ,
           ser.corr.warn = 0.99)

  {
    
    # Estimates the impulse response of y to one or more x's over lags from 0 to m, with optional case weights "wt", while correcting for AR noise of arbitrary order.
    # Tikhonov-Phillips regularization can be optionally applied by setting nu>0.
    
    # takes as input:
    # y               a vector or time series of a single response variable.  May contain NA's.
    #
    # x               a vector, matrix, or time series containing one or more inputs of same length as y.  May contain NA's.
    #
    # xnames          optional vector of strings that name each input.  Length must equal the number of columns of x
    #
    # wt              optional vector of case weights of same length as y.  May contain NA's.
    #
    # m               maximum lag, in number of time steps.  Number of lags will be m+1 (the extra 1 is for lag zero).
    #
    # nk              number of knots in piecewise linear broken stick representation of beta as a function of lag.
    #                           Must be integer greater than 2 and less than or equal to m+1, or must be zero (the default).  
    #                           If nk>2 and nk<=m+1, nk knots will be created at lags of 0, 1, and a geometric progression 
    #                           (or as close to geometric as it can be, given that the lags are integers) between lags 1 and m.
    #                           If nk<=2, nk>m+1, or nk==0 (the default), the broken-stick approach is not used, and instead 
    #                           the IRF is evaluated for m+1 lags between 0 and m.
    #
    # nk_buffer       flag for whether to buffer the broken-stick IRF by adding an extra knot to the geometric series beyond m,
    #                           and then remove that extra knot when reporting the results.  This approach addresses the artifactual
    #                           inflation of the IRF at the last knot by pushing that knot beyond the lags of interest and then ignoring 
    #                           its value.  Default is TRUE.  This functionality was not present in the originally published IRF code.
    #
    # nu              fractional weight, 0<=nu<1, to be given to Tikhonov-Phillips regularization (0 = no regularization)
    #
    # h               integer order of autoregressive correction (non-negative integer).  AR corrections of sufficiently high order can, 
    #                          by the duality principle, be used to account for moving average (MA) noise as well.  
    #                          The value of h should be much less than m; otherwise strange results may occur.  If h==0, no correction is applied.
    #
    #                          If h==NULL, the order of autoregressive correction will be determined automatically, based on both statistical significance
    #                          and practical significance.  The practical significance criterion tests whether the absolute values of all of the acf
    #                          and pacf coefficients are less than the user-specified threshold ARlim, below which they are assumed to be practically
    #                          insignificant (that is, to have no substantial effect on the estimates of the IRFs).  The statistical
    #                          significance criterion tests whether the acf and pacf coefficients are statistically distinguishable from white noise
    #                          at the specified significance level ARprob.  First abs(acf) and abs(pacf) are compared at each lag against the 
    #                          two-tailed p<0.05 critical value for correlation coefficients, 1.96/sqrt(n.eff), where n.eff is the effective sample size
    #                          that accounts for uneven weighting.  Then the number of cumulative exceedances at each lag L (e.g., the number of cases 
    #                          where abs(acf)>1.96/sqrt(n.eff) at lags from 1 to L) is be compared to the critical number of exceedances predicted by 
    #                          binomial statistics (qbinom(p=ARprob, size=L, prob=0.05)).  The residuals are considered white if the cumulative number of 
    #                          exceedances is less than or equal to this critical number, over all lags, in both the acf and the pacf.  h is increased
    #                          sequentially until either the practical significance criterion or the statistical significance criterion is met, 
    #                          or until h==max.AR (which triggers a warning).  The practical significance test is needed because in large samples, 
    #                          even trivially small acf and pacf coefficients may still be statistically significant, triggering a pointless effort to
    #                          make them smaller than they need to be.  Conversely, the statistical significance test is needed because in small samples,
    #                          even true white noise processes may yield acf and pacf coefficients that do not meet the practical significance threshold 
    #                          (that is, ARlim may be less than 1.96/sqrt(n.eff)), triggering a pointless effort to further whiten residuals that are 
    #                          already white.
    #
    # ARprob          significance threshold for testing whether residuals are sufficiently white in automatic AR selection
    #
    # ARlim           threshold value of acf and pacf coefficients of residuals, used in practical significance test
    #
    # max.AR          maximum order of autoregressive correction that will be accepted in automatic AR order selection
    #
    # complete        flag for whether the number of lags will be *assumed* to be sufficient to hold the complete IRF (TRUE), meaning that any IRF
    #                          coefficients at longer lags are trivially small, or whether this cannot be assumed (FALSE).  Complete==TRUE will yield
    #                          smaller, and more accurately estimated, standard errors if the real-world IRF actually does converge to nearly zero before 
    #                          the maximum lag is reached.  But if this is not the case, complete==TRUE will force the IRF to artifactually
    #                          converge toward zero at the longest lag (with artificially small standard errors).  Complete==TRUE should thus
    #                          be invoked with caution.
    #
    # verbose         controls whether progress reports are printed (TRUE) or suppressed (FALSE)
    #
    # robust         flag controlling whether robust estimation by Iteratively Reweighted Least Squares (IRLS) will be used
    #
    # max.chunk       maximum size, in bytes, of the largest piece of the regression matrix (xx, the matrix of x and its lags) that will be created
    #                          at one time.  Such xx matrices that would be larger than max.chunk will be created and processed
    #                          in separate "chunks" to avoid triggering memory paging, which could substantially increase runtime,
    #                          or to avoid exceeding the available virtual memory, which will lead to a crash.
    #                          Keeping chunk.max relatively small (order 1e8 or 1e9) incurs only a small performance penalty, *if* 
    #                          one does not perform robust estimation.  But in robust estimation, the xx matrix can be re-used if it is
    #                          built in one piece, whereas if it is chunked it will need to re-built several times.  Therefore users should
    #                          not set max.chunk much smaller than it really needs to be.
    #                          Setting max.chunk=NULL suppresses chunking entirely.
    # ser.corr.warn   warning threshold for lag-1 serial correlation of residuals.  When this is exceeded, a warning is issued suggesting that time steps should be aggregated.
    #                       Note that warning will only be issued if h is 0 or NULL; for other values of h this test would not have the same relevance.
    #                       ser.corr.warn is passed through from ERRA.
    
    
    # y, x, and wt can contain missing values.  Missing values of y create 1+h missing rows in the regression matrix.
    #                          Missing values of x create m+1 missing rows (one for each lag) in the regression matrix.
    
    # returns a list as output, with the following objects
    #
    # lags        vector of lags (in number of time steps)
    #
    # IRF         impulse response function for each explanatory variable x: matrix of ncol(x) columns and m+1 rows, corresponding to lags 0 through m) (or nk rows, corresponding to knots, if nk>0)
    #
    # se          standard errors of IRF coefficients: matrix of ncol(x) columns and m+1 rows, corresponding to lags 0 through m) (or nk rows, corresponding to knots, if nk>0)
    #
    # Kbb         covariance matrix of dimensions ncol(x)*(m+1) by ncol(x) by (m+1) (or ncol(x) by nk rows and columns, if nk>0)
    #
    # n           length of original x and y series
    #
    # n.eff       effective sample size, accounting for uneven weighting
    #
    # n.nz        number of nonzero values (that also have nonzero weight) in each explanatory variable x at each lag  (matrix of ncol(x) columns and m+1 rows, corresponding to lags 0 through m)
    #
    # h           order of AR correction that was applied
    #
    # phi         fitted AR coefficients (vector of length h)
    #
    # resid       residuals (vector of length n)
    #
    # resid.acf   autocorrelation function of residuals
    #
    # resid.pacf  partial autocorrelation function of residuals
    
    
    
    
    #//////////////////////////////////////
    # do preliminary checks and conversions
    
    if (verbose) TICK <- Sys.time()
    
    if (is.ts(x)) {                                  # if x is a time series, coerce it to a vector or matrix, as appropriate
      if (is.vector(x)) x <- as.numeric(x)           # a one-dimensional time series is easy
      else x <- matrix(as.numeric(x), ncol=NCOL(x))  # otherwise we need to coerce to a vector and then reassemble the matrix
    }
    
    if (is.ts(y)) y <- as.numeric(y)                 # if y is a time series, coerce it to a vector
    
    if ((!is.vector(x)) & (!is.matrix(x))) stop("Fatal error in IRF: x must either be a matrix or a vector")
    
    nx <- NCOL(x)                                    # nx is the number of x variables
    
    n <- NROW(x)                                     # n is number of time steps (NROW works for both vectors and matrices)
    if (length(y)!=n |
        length(wt)!=n ) stop("Fatal error in IRF: input series x, y, and wt are not all equal in length")
    
    if( nu<0 | nu>=1 ) stop("Fatal error in IRF: nu must be >=0 and <1")
    
    if( (round(m)!=m) | m<0 ) stop("Fatal error in IRF: m must be a non-negative integer")
    
    if ((complete!=TRUE) & (complete!=FALSE)) stop("Fatal error in IRF: complete must be TRUE or FALSE")
    
    if (!is.null(h))
      if ((round(h)!=h) | (h<0)) stop("Fatal error in IRF: h must be a non-negative integer, or NULL")
    
    if (ARprob<=0 | ARprob>=1 ) stop("Fatal error in IRF: ARprob must be between 0 and 1")
    
    if (max.AR>20) warning("Large max.AR in IRF")
    
    if(round(nk)!=nk) stop("Fatal error in IRF: nk must be an integer")
    if (nk!=0) {
      if (nk>=(m+1)) {
        nk <- 0
        warning("nk too big -- reverting to evenly spaced lags in IRF")
      } else if (nk<3) {
        nk <- 0
        warning("nk too small -- reverting to evenly spaced lags in IRF")
      } 
    }
    
    x <- as.matrix(x)                                # this converts x into a (potentially one-column) matrix, so that what follows will mostly not need to handle nx==1 and nx>1 as separate cases
    
    orig_y <- y                                      # save a copy of the original values of y, in case we do first-differencing
    
    
    #//////////////////////////////////
    # here we build the lag knot series (if nk>0)
    
    
    if (nk>0) {
      # knot series should start as 0, 1, and then follow geometric progression to reach m at the nk'th knot, but with a minimum step size of 1 (no duplicate knots!)
      if (nk_buffer==TRUE) knots <- rep(NA, nk+1) else knots <- rep(NA, nk)                             # declare knot vector
      knots[1] <- 0
      knots[2] <- 1
      for (k in 3:nk) {
        ratio <- (m/knots[k-1])^(1/(nk-(k-1)))             # ratio for a geometric progression that will reach value of m by knot number nk
        if ((ratio-1)*knots[k-1]>1) knots[k] <- round(knots[k-1]*ratio)    # follow geometric progression if this would be a step bigger than 1
        else knots[k] <- knots[k-1] + 1                                # otherwise take a step of 1
      }
      
      if (nk_buffer==TRUE){                                                  # add the buffer knot here
        if ((ratio-1)*knots[nk]>1) knots[nk+1] <- trunc(knots[nk]*ratio)     # follow geometric progression if this would be a step bigger than 1
        else knots[nk+1] <- knots[nk] + 1                                    # otherwise take a step of 1
      }
      
    } else {
      knots <- NULL
    }
    
    
    
    
    # determine whether we will need to do chunking or not
    byte.size <- 8       # nominal size (in bytes) of a numeric variable in R (not counting the one-time overhead, for example 48 bytes for any vector)
    # No need to change byte.size unless this fundamental property of R changes.  This is only used for calculating chunk sizes in solvAR.
    if (is.null(max.chunk)) chunk.maxht <- NULL else {
      if (nk>0) {
        if (nk_buffer==TRUE) chunk.maxht <- floor(max.chunk / ((nk+1)*nx* byte.size))  # maximum height of a chunk of xx matrix, to be passed to solvAR
        else chunk.maxht <- floor(max.chunk / (nk*nx* byte.size))  # maximum height of a chunk of xx matrix, to be passed to solvAR
      }
      else chunk.maxht <- floor(max.chunk / ((m+1)*nx* byte.size))  # maximum height of a chunk of xx matrix, to be passed to solvAR
    } 
    
    
    #////////////////////////////////////
    # here we start the AR selection loop
    # if autoAR==TRUE (i.e., h=NULL), this loop incrementally increases h until pacf.limit is met, or max.AR is reached
    # if autoAR==FALSE (i.e., h=integer), this loop runs just once at the user-supplied AR order
    
    
    if (is.null(h)) {
      h <- 0
      autoAR <- TRUE
    } else autoAR <- FALSE
    
    firstpass <- TRUE                  # flag for whether we are taking our first pass through the AR selection loop
    
    repeat {                           # start the AR selection loop
      
      n_lags <- m+h                    # total number of lags we will need (not counting lag zero)
      
      # the strategy for handling NA's is as follows: we assign a weight of 0 to any row that will contain NA's (also for any lagged variables)
      # and in solvAR we set NA's to zero (so the crossprod function doesn't choke), but these will correspond to rows with zero weight so the
      # numeric values will be preserved
      
      wt[1:n_lags] <- 0                                               # assign weight of zero to first n_lags rows (because these will contain NAs)
      wt[is.na(wt)] <- 0                                              # assign weight of zero to rows where weight is missing
      
      # if a row is missing, assign weight of zero to that row and the next n_lags rows (total of 1+n_lags rows)
      # this is the non-vector way we did it previously: for (i in 1:n) if (missing.row[i]) wt[i:min(n, i+n_lags)] <- 0 # need to do this because those rows would contain a lagged NA value of x
      missing.row <- as.integer(is.na(rowSums(x, na.rm=FALSE)))       # boolean vector of rows that have NA in x
      if (sum(missing.row)>0) {
        missing.cancels <- rep(1, 1+n_lags)                           # this is a mask that propagates the effects of missing x's forward n_lags steps
        missing.row <- as.vector(stats::filter(missing.row, missing.cancels, sides=1, circular=TRUE))  # propagate the mask
        wt[(missing.row>0.5)] <- 0                                    # and set weight for masked rows to zero (using >0.5 here in case we have any near-zero values.  We should have only integers, but let's not take chances.)
      }
      
      # if a value of y is missing, assign weight of zero to that row and the next h rows (total of 1+h rows) because AR terms (lagged y's) will be NA at those rows
      # this is the non-vector way we did it previously:  for (i in 1:n) if (is.na(y[i])) wt[i:min(n, i+h)] <- 0        # need to do this because those rows would contain a lagged NA value of y
      missing.y <- as.integer(is.na(y) )                              # boolean vector of rows that have NA in y
      if (sum(missing.y)>0) {
        if (h==0) wt[(missing.y>0)] <- 0
        else {
          missing.cancels <- rep(1, 1+h)                              # this is a mask that propagates the effects of missing y's forward h steps
          missing.row <- as.vector(stats::filter(missing.y, missing.cancels, sides=1, circular=TRUE))  # propagate the mask
          wt[(missing.row>0.5)] <- 0                                    # and set weight for masked rows to zero
        }
      }
      
      
      #//////////////////////////////////////////////
      # now do cross-products and solve linear system
      
      if (robust & firstpass) {  
        
        # If we are doing robust estimation, we first need to determine robustness weights, setting h=0 because robust estimation doesn't play well with AR correction.
        # Note that we need h=0 here even if we subsequently are using a different level of AR correction.
        if (nk>0) s1 <- BrokenStickRobustSolvAR(y=y, x=x, wt=wt, knots=knots, h=0, nu=nu, robust=robust, verbose=verbose)
        else s1 <- RobustSolvAR(y=y, x=x, wt=wt, m=m, h=0, nu=nu, robust=robust, verbose=verbose, chunk.maxht=chunk.maxht)
        
        # now multiply weights by robustness weights, and use these in all further trips around this loop.
        # Don't change wt vector after this, except for zeroing out additional values as needed if h increases!
        wt <- wt*s1$rw  
        
        firstpass <- FALSE                                                 # now turn firstpass flag off, so we won't keep coming back here
        
        if ((autoAR==FALSE) & (h==0)) {
          list2env(s1, envir=environment())                                # unpack list from solvAR
          break                                                            # and jump out of loop          
        }
        
      }  # end if (robust & firstpass)
      
      
      # Here is where we do most of the work.  Note that we set robust=FALSE even if we are doing robust estimation, because we have already determined robustness weights above.
      if (nk>0) s1 <- BrokenStickRobustSolvAR(y=y, x=x, wt=wt, knots=knots, h=h, nu=nu, robust=FALSE, verbose=verbose)
      else s1 <- RobustSolvAR(y=y, x=x, wt=wt, m=m, h=h, nu=nu, robust=FALSE, verbose=verbose, chunk.maxht=chunk.maxht)
      
      list2env(s1, envir=environment())                                # unpack list from solvAR
      
      
      # s1 includes
      # b.hat          regression coefficients (matrix of nx columns and m+1 rows, or nx columns and nk rows if nk>0)
      # se             standard errors of regression coefficients
      # Kbb            covariance matrix of regression coefficients (matrix of dimensions nx*(m+1) by nx*(m+1), or nx*nk by nx*nk if knots!=NULL)
      # allb           (only if nk>0) matrix of interpolated regression coefficients for each lag (matrix of nx columns and m+1 rows, which includes extra rows in nk_buffer==TRUE)
      # n              total number of time steps
      # phi            fitted AR coefficients
      # n.eff          effective number of time steps, accounting for uneven weighting
      # n.nz           a vector of mm+1 tallies of number of nonzero elements in the *weighted* x's (so values with zero weight don't count)
      # e              residuals
      # s2e            weighted residual variance
      # resid.acf      autocorrelation function of residuals
      # resid.pacf     partial autocorrelation function of residuals
      
      
      #///////////////////////////////////////////////
      # here we test for exiting the AR selection loop
      
      
      if ((h==0) & (resid.acf[1]>ser.corr.warn)) warning("Residuals have lag-1 serial correlation=",  round(resid.acf[1], 4), "  Consider aggregating time steps.")      
      
      if (autoAR==FALSE) {break}                                                                 # if we weren't doing automatic AR order selection, exit here
      
      
      if (verbose) cat("at h=", h, " and time=", round(difftime(Sys.time(), TICK, units="secs"), 3), " residual PACF (first 5 terms) = ", round(resid.pacf[1:5], 4), "\n")
      
      if (verbose & (h>0)) {
        cat("AR coefficients (phi) = ", s1$phi, "\n")
      }
      
      # first we do the practical significance test
      
      if ( max(abs(resid.pacf), abs(resid.acf[-1])) < ARlim ) {                                  # if we pass the practical significance test...
        if (verbose) cat(paste0("practical (in)significance test passed at h=", h, "\n"))        # write a note
        {break}                                                                                  # and exit
      }
      
      
      # The statistical significance test inevitably involves multiple comparisons (multiple acf and pacf values to be compared to a threshold).
      # It does not use a Bonferroni-style alpha value to account for these multiple comparisons, because such small alphas might be vulnerable to distributional anomalies in the residuals.
      # Instead we tally the number of times that we exceed the p<0.05 threshold for individual acf and pacf estimates, and then test whether this number of exceedances
      # is improbable at a significance level of ARprob.  These comparisons are made sequentially: first we check whether the number of exceedances at the first lag is greater than
      # the expected number (which, if ARprob=0.05, is zero), then we check whether the number of exceedances in the first two lags is greater than the expected number (which, 
      # if ARprob=0.05, is one), then we check over the first three lags, and so on out to the length of the acf or pacf.  
      # We apply this criterion separately to the acf and pacf, because AR noise will cause many exceedances in the acf (but not the pacf), 
      # whereas MA noise will cause many exceedances in the pacf (but not the acf).
      
      threshold <- 1.96/sqrt(n.eff)                                                              # this is the (absolute) value of an *individual* acf or pacf element that would be *individually* statistically significant at p=0.05
      acf.ncrit <- qbinom(p=ARprob, size=seq(length(resid.acf)-1), prob=0.05, lower.tail=FALSE)  # running tally of the critical number of abs(acf)>acf.limit that we would expect to occur <ARprob fraction of the time 
      pacf.ncrit <- qbinom(p=ARprob, size=seq(length(resid.pacf)), prob=0.05, lower.tail=FALSE)  # running tally of the critical number of abs(pacf)>acf.limit that we would expect to occur <ARprob fraction of the time
      acf.exceedances <- cumsum(abs(resid.acf[-1])>threshold)                                    # tally the running total number of exceedances in the acf
      pacf.exceedances <- cumsum(abs(resid.pacf>threshold))                                      # tally the running total number of exceedances in the pacf
      
      if ((sum(acf.exceedances>acf.ncrit)==0) & (sum(pacf.exceedances>pacf.ncrit)==0)) {         # if we meet both of the statistical criteria...
        if (verbose) cat(paste0("statistical (in)significance test passed at h=", h, "\n"))      # write a note
        {break}                                                                                  # and exit
      }
      
      
      if (h >= max.AR) {                                                                         # if we have reached the maximum AR order allowed
        warning(paste0("Maximum AR order of ", max.AR, " reached in iterative loop. Consider aggregating time steps or first differencing."))            # print a warning
        {break}                                                                                  # and exit
      }
      
      h <- h + 1                                                                                 # if we haven't exited, increment h
      
    } # inner (AR selection) loop ends here                                                      # and continue around the AR selection loop again
    
    
    
    
    
    
    #///////////////////////////////////////////////////////////////////////////////////////////////
    # Now we correct the coefficients and their standard errors for AR noise
    # We use one of two different ways, depending on whether the IRF is assumed to be complete or not
    
    if (h>0) {                                                        # if h==0, then the first-stage results (s1) are already complete and we can go straight to the end
      
      #/////////////////////////////////////////////
      if (complete==FALSE) {                                          # if the IRF is not assumed complete, then...:
        
        # now we need to adjust the IRF coefficients and their standard errors to take account of the autoregressive noise
        # to do this we need to construct the psi matrix (the inverse of phi)
        
        if (nk==0) {    # if we have evenly spaced lags (conventional IRF, not broken-stick IRF)
          
          v <- c(1, (-1*s1$phi), rep(0, m-h))                           # start with a vector whose first element is 1, followed by the phi values, and padded out with zeros to length m+1  (equation 20 of K2022)
          phi.mat <- toeplitz(v)                                        # create the corresponding Toeplitz matrix (symmetric, for now)
          phi.mat[upper.tri(phi.mat)] <- 0                              # and zero out the upper triangle... voila!
          psi <- solve(phi.mat)                                         # and take the inverse
          
          # if we have more than one x-variable, we need a block-diagonal psi matrix
          if (nx>1) psi <- as.matrix(bdiag(replicate(nx, psi, simplify=FALSE)))
          
          # note that after this block, b.hat and se will be vectors instead of matrices
          b.hat <- psi %*% as.vector(b.hat)                             # convert b's using the inverse of the phi matrix   (equation 24 of K2022)
          Kbb <- psi %*% Kbb %*% t(psi)                                 # use psi to propagate the covariance matrix (this is the standard way to do it, since psi is the Jacobian for the translation of the b's)  (equation 26 of K2022)
          se <- sqrt(diag(Kbb))                                         # and the square root of the diagonal gives the standard errors
          
          
        } else {    # if we have a broken-stick IRF
          
          v <- c(1, (-1*s1$phi), rep(0, knots[length(knots)]-h))        # start with a vector whose first element is 1, followed by the phi values, and padded out with zeros to maximum lag  (equation 20 of K2022)
          phi.mat <- toeplitz(v)                                        # create the corresponding Toeplitz matrix (symmetric, for now)
          phi.mat[upper.tri(phi.mat)] <- 0                              # and zero out the upper triangle... voila!
          psi <- solve(phi.mat)                                         # and take the inverse
          
          # if we have more than one x-variable, we need a block-diagonal psi matrix
          if (nx>1) psi <- as.matrix(bdiag(replicate(nx, psi, simplify=FALSE)))
          
          # note that here, allb will become a vector instead of a matrix
          allb <- psi %*% as.vector(allb)                              # convert allb's using the inverse of the phi matrix   (equation 24 of K2022)
          # b.hat <- allb[(knots+1)]                                     # sample the allb's at the knot points to retrieve the AR-corrected beta values (need the +1 because lag zero is index=1...)
          # if (nx>1) {
          #   maxknot <- max(knots)               
          #   for (j in 2:nx) b.hat <- c(b.hat, allb[knots+1+((j-1)*maxknot)])  # need to keep sampling successive knot points in the allb vector if we have multiple x's
          # }
          
          allb <- matrix(allb, ncol=nx)                                # make allb a matrix again
          
          b.hat <- as.vector(allb[(knots+1), , drop=FALSE])
          
          # Unfortunately the procedure above does not allow us to correct the standard error estimates for the autocorrelation in the noise.
          # For that, we will do something analogous to Cochrane-Orcutt, 
          # but without iterating to find the phi values -- we already have them
          new_y <- y
          for (i in 1:h) new_y <- new_y - s1$phi[i]*dplyr::lag(y, i)    # transform y to remove autoregressive errors (based on phi from first-stage results)
          
          new_x <- x                                                    
          for (i in 1:h) new_x <- new_x - s1$phi[i]*dplyr::lag(x, i)    # transform x to match the transformation of y                
          
          # now call BrokenStickSolvAR again, *without* AR terms, using AR-corrected x *AND* y, to get the standard error estimates (and new IRF coefficients, *if* we assume that the IRF is "complete"
          s2 <- BrokenStickRobustSolvAR(y=new_y, x=new_x, wt=wt, knots=knots, h=0, nu=nu, verbose=verbose, robust=FALSE)   # call solvAR again, now *without* AR terms, to get new IRF coefficients
          
          se <- s2$se                                                   # copy the standard errors from this run
          Kbb <- s2$Kbb                                                 # copy the parameter covariance matrices too
          
        } # end if (nk==0)
        
      } else {           # if complete==TRUE                                             
        
        # if we assume the IRF is complete, we will do something analogous to Cochrane-Orcutt, 
        # but without iterating to find the phi values -- we already have them
        new_y <- y
        for (i in 1:h) new_y <- new_y - s1$phi[i]*dplyr::lag(y, i)    # transform y to remove autoregressive errors (based on phi from first-stage results)
        
        new_x <- x                                                    
        for (i in 1:h) new_x <- new_x - s1$phi[i]*dplyr::lag(x, i)    # transform x to match the transformation of y                
        
        # now call solvAR again, *without* AR terms, using AR-corrected x *AND* y, to get new IRF coefficients
        if (nk>0) s2 <- BrokenStickRobustSolvAR(y=new_y, x=new_x, wt=wt, knots=knots, h=0, nu=nu, verbose=verbose, robust=FALSE)          # call solvAR again, now *without* AR terms, to get new IRF coefficients
        else s2 <- RobustSolvAR(y=new_y, x=new_x, wt=wt, m=m, h=0, nu=nu, verbose=verbose, robust=FALSE, chunk.maxht=chunk.maxht)  # call solvAR again, now *without* AR terms, to get new IRF coefficients
        
        list2env(s2, envir=environment())                             # unpack list from solvAR
        
      } # end else                                                   # and that's it!  we don't need to transform beta's or s.e.'s in this case
      
    } # end if (h>0)
    
    if (verbose & (nk==0)) cat("\n")
    if (verbose) cat("IRF finished...:", round(difftime(Sys.time(), TICK, units="secs"), 3), " seconds\n")
    
    if (verbose) cat(paste0("minimum column count of nonzero x values = ", min(n.nz), " (", round(100*min(n.nz)/max(n.nz)), "% of maximum count, and ", round(100*min(n.nz)/sum(wt>0), 1), "% of rows with nonzero weight)\n"))
    
    if (verbose & (h>0)) {
      cat("AR coefficients (phi) = ", s1$phi, "\n")
    }
    
    if (verbose) {
      cat("residual PACF (first 5 terms) =", round(resid.pacf[1:5], 4), "\n")
    }
    
    
    #//////////////////////////////////////////////
    # Now calculate predicted values of y from the AR-adjusted coefficients, to see how well the statistical model fits the time-series behavior of y.
    # That may not be realistically reflected by the residuals in solvAR, because if y is strongly autocorrelated (and h>0), values of y will inherently
    # be close to their previous values.  This artifact is avoided if we use AR-adjusted coefficients and *only* x, and not lagged values of y,
    # to predict y.
    
    # first we make a naive estimate of y, without an intercept (because this has been lost in the AR correction)
    if (nk>0) {
      if (nk_buffer==TRUE) allb <- allb[ -(m+2):-(max(knots)+1), ]    # need to remove rows that were added to extend to the buffer knot
      ypred <- calc_ypred(x=as.vector(x), beta=as.vector(allb), nx=NCOL(x), n=NROW(x), m=m, h=h)
    } else {
      ypred <- calc_ypred(x=as.vector(x), beta=as.vector(b.hat), nx=NCOL(x), n=NROW(x), m=m, h=h)
    }
    
    ww <- wt[(n_lags+1):n]
    
    # The next two lines are deprecated, because in poorly fitted models they lead to constant nonzero streamflow when p=0.
    # then to add the intercept, we determine the weighted average difference between ypred and y, with robustness weights if robust==TRUE    
    # ypred <- ypred + (weightedMean(orig_y[(n_lags+1):n], ww, na.rm=TRUE) - weightedMean(ypred, ww, na.rm=TRUE))
    
    ycomp <- data.table(timestep=(n_lags+1):n, wt=ww, y=orig_y[(n_lags+1):n], ypred=ypred, yresid=(orig_y[(n_lags+1):n]-ypred), x=x[(n_lags+1):n,])
    
    if ((nk_buffer==TRUE) && (nk>0)) {       # if we added a buffer knot...
      knots <- knots[1:nk]                   # remove it from the list of knots
      keep.mask <- as.logical(rep(c(rep(1, nk), 0), nx)) # this is a mask for values to keep
      b.hat <- b.hat[keep.mask]              # remove the corresponding elements of b.hat
      se <- se[keep.mask]                    # and remove the corresponding elements of se
      Kbb <- Kbb[ keep.mask, keep.mask]      # remove corresponding columns from Kbb
    }
    
    if (nk>0) lags <- knots 
    else lags <- 0:m
    
    b.hat <- matrix(as.vector(b.hat), ncol=nx)  # wrap these back to matrices
    se <- matrix(as.vector(se), ncol=nx)
    
    # now create column names
    # if vector of xnames does not exist, create it
    if (is.null(xnames) | length(xnames)!=nx) xnames <- paste0("x", 1:nx)
    
    colnames(b.hat) <- paste0("IRF_", xnames)
    colnames(se) <- paste0("se_", xnames)
    
    
    return(
      list(
        lags = lags ,      # lags (in number of time steps)
        IRF = b.hat ,       # impulse response function for each column of x matrix: nx columns and m+1 rows (or nk rows, if nk>0)
        se = se ,            # standard errors: nx columns and m+1 rows (or nk rows, if nk>0)
        Kbb = Kbb ,           # covariance matrix of IRF (dimensions nx*(m+1) by nx(m+1), or nx*nk by nx*nk if nk>0) 
        n = n ,                # length of original x and y series
        n.eff = s1$n.eff ,      # effective sample size, accounting for uneven weighting
        n.nz = s1$n.nz ,         # number of nonzero values with nonzero weight at each lag of each x variable
        h = h ,                   # order of AR correction that was applied
        phi = s1$phi ,             # fitted AR coefficients (vector of length h)
        resid = e ,                 # residuals
        resid.acf = resid.acf ,      # autocorrelation function of residuals
        resid.pacf = resid.pacf ,     # partial autocorrelation function of residuals
        ycomp = ycomp                  # data table comparing measured and fitted y time series (starting after initial lags)
        
      )
    ) # end return
    
    
    
  }  #end IRF

#///////////////////
# END of IRF
#///////////////////










#///////////////////////////////////////////////////////////////////////////////////////////////
#### nonlinIRF - nonlinear Impulse Response Function estimates with correction for AR noise ####
#///////////////////////////////////////////////////////////////////////////////////////////////


nonlinIRF <- function(y ,
                      x ,
                      xnames = NULL ,
                      wt = rep(1, NROW(x)) ,
                      m = 60 ,
                      nk = 0 ,
                      xknots = c(20, 50, 80, 90, 95) ,
                      pct_xknots = TRUE ,
                      nu = 0 ,
                      h = NULL ,
                      ARprob = 0.05 ,
                      ARlim = 0.2 ,
                      max.AR = 12 ,
                      complete = FALSE ,
                      verbose = FALSE ,
                      robust = FALSE ,
                      max.chunk = 2e8 ,
                      ser.corr.warn = 0.99)
{
  # Shell that calls IRF to estimate nonlinear impulse response functions, i.e., IRFs that depend on the input (x)
  
  
  # takes as input:
  # y               a vector or time series of a single response variable.  May contain NA's.
  #
  # x               a vector, matrix, or time series containing one or more inputs of same length as y.  Must be non-negative.  May contain NA's.
  #
  # xnames          optional vector of strings that name each input.  Length must equal the number of columns of x
  #
  # xknots          a vector or matrix of knots for the piecewise linear approximation of the RRD's nonlinear dependence on x.
  #                       Knots can be specified as fixed values or as percentiles of the x distribution (depending on the pct_knots flag -- see below).
  #                                  Values of p=0 are ignored when these percentiles are later converted to fixed values
  #                       If xknots is a matrix, it must have ncol(x), and each column of knots will be applied to the corresponding column of x.
  #                                  Each column of knots must have the same number of values, although the xknot values themselves may differ.
  #                       If xknots is a vector or single-column matrix and x is a multi-column matrix, the same xknots will be applied to each column of x.
  #                       Xknots must be between (and not include) the minimum and maximum values of the corresponding column of x.  
  #                       If xknots == NULL, potential nonlinear responses to input intensity are ignored and a single IRF is estimated for each 
  #                                 column of x.
  #                       If xknots != NULL, separate IRF's are estimated for each specified knot point (and also the maximum value) of each column of x.
  #
  # pct_xknots      a flag indicating whether nonlinearity knots are expressed values of x (FALSE) or percentiles (TRUE, 1, 2, or 3).
  #                       If pct_xknots==TRUE or pct_xknots==1, xknots are calculated as percentiles of x.
  #                       If pct_xknots==2, xknots are calculated as percentiles of the cumulative sum of x (so if xknots=80, for example, the xknot will be the value of x 
  #                                 for which all smaller x's sum to 80 percent of the sum of all x's).  Thus these xknots will delimit fractions of the total input x.
  #                       If pct_xknots==3, xknots are calculated as percentiles of the cumulative sum of squares of x (so if xknots=20, for example, the xknot will be the value of x for which all smaller x's, 
  #                                 squared and summed, add up to 20 percent of the sum of squared x's).  Thus these xknots will delimit fractions of the total sum of squared inputs x^2.
  #                                 These will roughly approximate the corresponding fractions of the total leverage in the data set, if the distribution of x's is strongly skewed with a peak near zero and a long right tail.
  #                       Any values other than TRUE, FALSE, 0, 1, 2, or 3 will trigger an error.
  #
  #                       xknots should be chosen so that (these are not checked):
  #                       (a) there are enough points in each interval between xknots, and sufficient variability in their values, to define the dependence of y on x,
  #                       (b) intervals between individual pairs of xknots do not span major changes in slope in the nonlinear relationship between y and x
  #  
  #                       y, x, and wt can contain missing values.  Missing values of y create 1+h missing rows in the design matrix.
  #                       Missing values of x create m+1 missing rows (one for each lag) in the regression matrix.
  #
  # ser.corr.warn   warning threshold for lag-1 serial correlation of residuals.  When this is exceeded, a warning is issued suggesting that time steps should be aggregated.
  #                       Note that warning will only be issued if h is 0 or NULL; for other values of h this test would not have the same relevance.
  #                       ser.corr.warn is passed through from ERRA, and passed along to IRF.
  #                        
  #
  #
  #
  # The remaining inputs are identical to those in IRF:
  #
  # wt              optional vector of case weights of same length as y.  May contain NA's.
  #
  # m               maximum lag, in number of time steps.  Number of lags will be m+1 (one extra for lag zero).
  #
  # nk              number of knots in piecewise linear broken stick representation of beta as a function of lag.
  #                           Must be integer greater than 2 and less than or equal to m+1, or must be zero (the default).  
  #                           If nk>2 and nk<=m+1, nk knots will be created at lags of 0, 1, and a geometric progression 
  #                           (or as close to geometric as it can be, given that the lags are integers) between lags 1 and m.
  #                           If nk<=2, nk>m+1, or nk==0 (the default), the broken-stick approach is not used, and instead 
  #                           the IRF is evaluated for m+1 lags between 0 and m.
  #
  # nu              fractional weight, 0 <= nu < 1, to be given to Tikhonov-Phillips regularization (0 = no regularization)
  #
  # h               integer order of autoregressive correction (non-negative integer).  AR corrections of sufficiently high order can, 
  #                          by the duality principle, be used to account for moving average (MA) noise as well.  
  #                          The value of h should be much less than m; otherwise strange results may occur.  If h==0, no correction is applied.
  #
  #                          If h==NULL, the order of autoregressive correction will be determined automatically, based on both statistical significance
  #                          and practical significance.  The practical significance criterion tests whether the absolute values of all of the acf
  #                          and pacf coefficients are less than the user-specified threshold ARlim, below which they are assumed to be practically
  #                          insignificant (that is, to have no substantial effect on the estimates of the IRFs).  The statistical
  #                          significance criterion tests whether the acf and pacf coefficients are statistically distinguishable from white noise
  #                          at the specified significance level ARprob.  First abs(acf) and abs(pacf) are compared at each lag against the 
  #                          two-tailed p<0.05 critical value for correlation coefficients, 1.96/sqrt(n.eff), where n.eff is the effective sample size
  #                          that accounts for uneven weighting.  Then the number of cumulative exceedances at each lag L (e.g., the number of cases 
  #                          where abs(acf)>1.96/sqrt(n.eff) at lags from 1 to L) is compared to the critical number of exceedances predicted by 
  #                          binomial statistics (qbinom(p=ARprob, size=L, prob=0.05)).  The residuals are considered white if the cumulative number of 
  #                          exceedances is less than or equal to this critical number, over all lags, in both the acf and the pacf.  h is increased
  #                          sequentially until either the practical significance criterion or the statistical significance criterion is met, 
  #                          or until h==max.AR (which triggers a warning).  The practical significance test is needed because in large samples, 
  #                          even trivially small acf and pacf coefficients may still be statistically significant, triggering a pointless effort to
  #                          make them smaller than they need to be.  Conversely, the statistical significance test is needed because in small samples,
  #                          even true white noise processes may yield acf and pacf coefficients that do not meet the practical significance threshold 
  #                          (that is, ARlim may be less than 1.96/sqrt(n.eff)), triggering a pointless effort to further whiten residuals that are 
  #                          already white.
  #
  # ARprob          significance threshold for testing whether residuals are sufficiently white in automatic AR selection
  #
  # ARlim           threshold value of acf and pacf coefficients of residuals, used in practical significance test
  #
  # max.AR          maximum order of autoregressive correction that will be accepted in automatic AR order selection
  #
  # complete        flag for whether the number of lags will be *assumed* to be sufficient to hold the complete IRF (TRUE), meaning that any IRF
  #                          coefficients at longer lags are trivially small, or whether this cannot be assumed (FALSE).  Complete==TRUE will yield
  #                          smaller, and more accurately estimated, standard errors if the real-world IRF actually does converge to nearly zero before 
  #                          the maximum lag is reached.  But if this is not the case, complete==TRUE will force the IRF to artifactually
  #                          converge toward zero at the longest lag (with artificially small standard errors).  Complete==TRUE should thus
  #                          be invoked with caution.
  #
  # robust          flag controlling whether robust estimation by Iteratively Reweighted Least Squares (IRLS) will be used
  #
  # verbose         controls whether progress reports are printed (TRUE) or suppressed (FALSE)
  #
  # max.chunk       maximum size, in bytes, of the largest piece of the design matrix (the matrix of x and its lags) that will be created
  #                          at one time.  Design matrices that would be larger than max.chunk will be created and processed
  #                          in separate "chunks" to avoid triggering memory paging, which could substantially increase runtime.  
  #                          Keeping chunk.max relatively small (order 1e8 or 1e9) incurs only a small performance penalty, except in the case of robust estimation,
  #                          where the need to iterate means that the solution will be faster (by about a factor of 2 or so) *if* one can keep the whole design matrix
  #                          in one chunk without triggering memory paging.
  
  
  # returns a list as output, with the following objects
  #
  # knots       knots, in matrix of nk rows and nx columns (not including each x's first knot, which is zero)
  #
  # IRF         nonlinear impulse response function (beta) evaluated for each x variable at each knot point (except zero): matrix of nx*nk columns and m+1 rows, corresponding to lags 0 through m)
  #
  # se          standard errors of IRF coefficients (matrix of nx*nk columns and m+1 rows, corresponding to lags 0 through m)
  #
  # ykx         nonlinear impulse response, expressed as contribution to y from x, evaluated at each knot point (except zero): matrix of nx*nk columns and m+1 rows, corresponding to lags 0 through m)
  #
  # ykx_se      standard errors of ykx (matrix of nx*nk columns and m+1 rows, corresponding to lags 0 through m)
  #
  #             Note that ykx will show the shape of the nonlinearity more intuitively than IRF.
  #             For a linear system, ykx will be a straight line, whereas IRF will be a constant (within uncertainty)
  #             For a quadratic dependence, ykx will be parabolic, whereas IRF will be a straight line
  #
  # avg_IRF     arithmetic average of IRFs over all time steps with nonzero x's (use this when IRFs converge toward zero as x approaches zero)
  #
  # avg_se      standard error of avg_IRF
  #
  # avg_ykx     time-averaged ykx (including time steps with x=0)
  #
  # avg_ykx_se  standard error of avg_ykx
  #
  # wtd_avg_IRF weighted average of IRFs (weighted by input x)
  #
  # wtd_avg_se  standard error of wtd_avg_IRF
  # 
  # wtd_avg_ykx weighted average of ykx (weighted by the input x)
  #
  # wtd_avg_ykx_se  standard error of wtd_avg_ykx
  #
  # seg_wtd_meanx  weighted mean of x in segments between pairs of knots
  #
  # seg_wtd_IRF weighted average of IRFs in segments between pairs of knots
  # 
  # seg_wtd_se  standard error of seg_wtd_IRF
  #
  # seg_wtd_ykx weighted average of ykx in segments between pairs of knots
  #
  # seg_wtd_ykx_se  standard error of seg_wtd_ykx
  #
  #
  #
  # the rest of these outputs are passed from IRF
  #
  # Kbb         stack of covariance matrices for each xprime (array of nx*nk matrices, each m+1 by m+1)
  #
  # n           length of original x and y series
  #
  # n.eff       effective sample size, accounting for uneven weighting
  #
  # n.nz        number of nonzero values (that also have nonzero weight) in each explanatory variable x at each lag  (matrix of ncol(x)*nk columns and m+1 rows, corresponding to lags 0 through m)
  #
  # h           order of AR correction that was applied
  #
  # phi         fitted AR coefficients (vector of length h)
  #
  # resid       residuals (vector of length n)
  #
  # resid.acf   autocorrelation function of residuals
  #
  # resid.pacf  partial autocorrelation function of residuals
  
  
  #############################
  # potential confusion alert!!
  #
  # Here we use broken-stick piecewise linear approximations for two different purposes.  We use one broken-stick model
  # to approximate the nonlinear dependence of y on x, and another to approximate the dependence of the IRF on lag time
  # (thus allowing users to capture both the rapid changes in the IRF at short lag times, and the slow changes in the
  # IRF at long lag times, without needing to estimate IRF coefficients for huge numbers of individual lags).
  #
  # This is potentially confusing because both of these broken-stick approximations have knots.  To disambiguate these 
  # two types of knots, we use "xknots" to refer to the knots in the broken-stick representation of y's dependence on x,
  # and "knots" to refer to the knots in the broken-stick representation of the relationship between the IRF and
  # lag time.  The corresponding numbers of these knots are "nxk" and "nk".  
  
  
  
  
  # check range of x
  if (min(x, na.rm=TRUE)<0) stop("Fatal error in nonlinIRF: x's may not be negative")
  
  
  # if x is not a matrix, make it one
  x <- as.matrix(x)
  nx <- ncol(x)            # number of x variables
  
  
  
  # prepare xknots vector
  # if nonlinearity knots are specified as a vector, make it a matrix with the same number of columns as x
  xknots <- as.matrix(xknots)
  if (ncol(xknots)!=nx) {
    if (ncol(xknots)!=1) stop(paste0("Fatal error in nonlinIRF: ", ncol(xknots), " knot columns but ", nx, "x columns"))
    else xknots <- matrix(data=xknots, nrow=length(xknots), ncol=nx)   # if only a single column of xknots is supplied, then copy it for every column of x's
  }
  
  nxk <- NROW(xknots)+1                                           # number of knots above zero, for which nonlinear response will need to be estimated
  
  xprime <- matrix(data=0, nrow=NROW(x), ncol=nx*nxk)             # this is the x_prime matrix (matrix of increments of x between each pair of knots)
  
  for (i in 1:nx) sort(xknots[,i])                                # sort into ascending order
  
  kpts <- matrix(data=0, nrow=nrow(xknots)+2, ncol=ncol(xknots))  # temporary matrix that we will use to prepare knots
  for (i in 1:nx) {
    
    xx <- x[,i]
    kp <- xknots[,i]
    
    if (pct_xknots!=FALSE) {
      
      xx <- xx[!is.na(xx)]                                 # discard nonzero x's and get rid of na's
      xx <- sort(xx[(xx>0)])                               # discard zeroes and sort remaining values in ascending order 
      
      if (pct_xknots==1) kp <- quantile(xx, probs=kp/100, na.rm=TRUE)   # convert percentile knots to values of x (note that these are quantiles of *nonzero* x's...)
      
      else if ((pct_xknots==2) | (pct_xknots==3)) { 
        # if pct_xknots==2, we set the knots according the percentile of the *running sum* of the x distribution.  That is, xknots=50 should find the value of X for which sum(x<X) equals sum(x>X).
        # if pct_xknots==3, we set the knots according the percentile of the *running sum* of the *squares* of the x distribution.  That is, xknots=50 should find the value of X for which sum((x<X)^2) equals sum((x>X)^2).
        
        xs <- cumsum(xx^(pct_xknots-1))                    # xs is the cumulative sum of the sorted values of xx (after they have been raised to the appropriate power)
        for (j in 1:length(kp)) kp[j] <- xx[which.min( abs(xs - max(xs)*kp[j]/100) )]    # this picks the values of xx that most closely delimit the corresponding fractions of the cumulative sum (or sum of squares if pct_xknots=3)
        
      } else stop("Fatal error in nonlinIRF: pct_xknots must be TRUE, FALSE, or an integer >=0 and <=3")
      
    }
    
    
    minx <- min(xx, na.rm=TRUE)
    maxx <- max(xx, na.rm=TRUE)
    
    if (sum(kp<=minx)>0) stop(paste0("Fatal error in nonlinIRF: one or more knots is <= minimum value of x for column ", i))
    if (sum(kp>=maxx)>0) stop(paste0("Fatal error in nonlinIRF: one or more knots is >= maximum value of x for column ", i))
    
    kpts[,i] <- c(0, kp, maxx)                # add knots of zero and max(x)
  } 
  
  xknots <- kpts # copy back to knots matrix
  
  if (sum(is.na(xknots))>0) stop("Fatal error in nonlinIRF: one or more knots is NA")
  
  seg_wtd_meanx <- matrix(data=0, nrow=nxk, ncol=nx)         # x-weighted average of x within each interval between knots
  seg_wtd_meanxprime <- matrix(data=0, nrow=nxk, ncol=nx)    # x-weighted average xprime corresponding to seg_wtd_meanx
  
  # here we expand each column of x's into columns of xprimes (to detect nonlinear dependence on x)
  for (i in 1:ncol(x)) {
    
    xx <- x[,i] # take each column of x separately
    kp <- xknots[,i] # take each column of knots separately
    
    # now write the appropriate columns of the matrix of xprimes (increments of x between knots)
    
    for (el in 1:nxk) {                                                  # calling this "el" instead of l so that it is not confused with 1
      
      interval <- kp[el+1]-kp[el]                                        # interval between knots
      
      xprime[ ,(el+(i-1)*nxk)] <- pmax(0, pmin(xx-kp[el], interval))     # this should give the correct increment values within each interval between knots (equation 43 of K2022)
      
      seg_wtd_meanx[el,i] <- sum(xx[(xx<=kp[el+1])&(xx>kp[el])]^2, na.rm=TRUE) / sum(xx[(xx<=kp[el+1])&(xx>kp[el])], na.rm=TRUE) # x-weighted mean of x within each interval   
      
      seg_wtd_meanxprime[el,i] <- seg_wtd_meanx[el,i] - kp[el]           # xprime corresponding to seg_wtd_meanx   
    }
  }
  
  
  
  ##############################################################
  # now call IRF with xprime, passing along all other parameters
  # since we are supplying xprimes here, the IRF routine will return beta-primes instead of betas
  
  zz <- IRF(y = y ,                                            
            x = xprime ,
            wt = wt ,
            m = m ,
            nk = nk ,
            nu = nu ,
            h = h ,
            ARprob = ARprob ,
            ARlim = ARlim ,
            max.AR = max.AR ,
            complete = complete ,
            verbose = verbose ,
            robust = robust ,
            max.chunk = max.chunk,
            ser.corr.warn = ser.corr.warn)
  
  
  #############################################################################################
  # now we need to convert betaprime to beta and system response y_k(x), with error propagation
  
  betaprime <- as.vector(zz$IRF)                                   # zz$IRF is beta-prime, because we supplied xprime rather than x to IRF.  Unpack this as a vector.
  # this will be a vector of length nx*nxk*(m+1), or nx*nxk*nk if nk>0
  Kbb <- zz$Kbb                                                    # this is the covariance matrix of beta-prime
  
  if (nk>0) nbeta <- nk                                            # nbeta is the number of coefficients for each x and each xknot (the total number of coefficients is nx*nxk*nbeta)
  else nbeta <- m+1
  
  
  
  # calculate averages of xprimes
  avg_xprime <- colMeans(xprime[complete.cases(xprime), ])   # averages of xprimes (excluding incomplete cases)
  
  # to calculate weighted averages of xprimes, we need to disaggregate by possible x's
  compx <- x[complete.cases(xprime), , drop=FALSE]  # x's for which we have all the xprimes
  compxprime <- xprime[complete.cases(xprime), , drop=FALSE]  # xprimes for which rows are complete
  
  
  # now we calculate weighted xprimes (for weighted average responses)
  for (j in 1:nx) {                               # step through each of the x's
    firstcol <- 1+(j-1)*nxk
    lastcol <- j*nxk
    wxprime <- colWeightedMeans(compxprime[, firstcol:lastcol], w=compx[, j])
    
    if (j==1) wtd_xprime <- wxprime 
    else wtd_xprime <- c(wtd_xprime, wxprime)
  }
  
  # now fold these into matrices, with all the avg_xprimes for a given x variable in one row
  avg_xprime <- matrix(avg_xprime, nrow=nx, byrow=TRUE)
  wtd_xprime <- matrix(wtd_xprime, nrow=nx, byrow=TRUE)
  
  
  # now construct matrices of coefficients for converting betaprimes to ykx's (NRF's at each knot)
  
  for (j in 1:nx) {                                # select each x variable
    dx <- rep(NA, nxk)                             # distances between knot values
    dx[1] <- xknots[2,j]                           # remember the first xknot is zero!
    for (i in 2:nxk) dx[i] <- xknots[(i+1),j]-xknots[i, j]  # remember the first xknot is zero!
    
    dxm.j <- matrix(0, nrow=nxk, ncol=nxk)         # blank matrix of dx's for jth x variable
    for (k in 1:nxk) dxm.j[k, 1:k] <- dx[1:k]      # replace first k values of kth row with dx's
    
    swdxm.j <- dxm.j                               # matrix of segment-weighted dx's for jth x variable
    diag(swdxm.j) <- seg_wtd_meanxprime[ ,j]       # replace diagonal with segment-weighted xprimes
    
    if (j==1) {
      dxm <- dxm.j                                 # dxm is xprimes at each knot, in a matrix form that will convert betaprime (vector of length nx*nxk) to ykx (vector of length nx*nxk)
      swdxm <- swdxm.j                             # swdxm is segment-weighted xprimes (differing only in the last nonzero value), in a matrix form that will convert betaprime (vector of length nx*nxk) to ykx (vector of length nx*nxk)
      adxm <- avg_xprime[j, , drop=FALSE]          # adxm is averaged xprimes, in a matrix form that will convert betaprime (vector of length nx*nxk) to average ykx (vector of length nx)
      wadxm <- wtd_xprime[j, , drop=FALSE]         # wadxm is weighted average xprimes, in a matrix form that will convert betaprime (vector of length nx*nxk) to average ykx (vector of length nx)
    } else {
      dxm <- bdiag(dxm, dxm.j)                     # combine these as block-diagonal matrices
      swdxm <- bdiag(swdxm, swdxm.j)
      adxm <- bdiag(adxm, avg_xprime[j, , drop=FALSE])                    
      wadxm <- bdiag(wadxm, wtd_xprime[j, , drop=FALSE])
    }
  } 
  
  dxm <- as.matrix(dxm)                            # convert these to regular matrices (may not be necessary)
  swdxm <- as.matrix(swdxm)
  adxm <- as.matrix(adxm)
  wadxm <- as.matrix(wadxm)
  
  lt <- rep(1:nbeta, nx*nxk)                                          # this cycles from 1:nbeta, nk*nxk times, identifying the lag times in the rows and columns of the Kbb matrix
  
  ykx <- matrix(NA, nrow=nx*nxk, ncol=nbeta)                          # this will hold the ykx's, in transposed order for now
  ykx_se <- matrix(NA, nrow=nx*nxk, ncol=nbeta)                       # and their standard errors, in transposed order for now
  seg_wtd_ykx <- matrix(NA, nrow=nx*nxk, ncol=nbeta)                  # this will hold the segment-weighted ykx's, in transposed order for now
  seg_wtd_ykx_se <- matrix(NA, nrow=nx*nxk, ncol=nbeta)               # and their standard errors, in transposed order for now
  avg_ykx <- matrix(NA, nrow=nx, ncol=nbeta)                          # this will hold the averaged ykx's, in transposed order for now
  avg_ykx_se <- matrix(NA, nrow=nx, ncol=nbeta)                       # and their standard errors, in transposed order for now
  wtd_avg_ykx <- matrix(NA, nrow=nx, ncol=nbeta)                      # this will hold the precipitation-weighted average ykx's, in transposed order for now
  wtd_avg_ykx_se <- matrix(NA, nrow=nx, ncol=nbeta)                   # and their standard errors, in transposed order for now
  
  for (k in 1:nbeta){                                                 # loop through each lag time
    bp <- betaprime[(lt==k)]                                          # select the betaprimes corresponding to this lag time
    bb <- Kbb[(lt==k) , (lt==k)]                                      # select the covariance matrix that corresponds to this lag time

    ykx[,k] <- dxm %*% bp                                             # each column of holds the ykx's for one lag time
    ykx_se[,k] <- sqrt(diag( dxm %*% bb %*% t(dxm) ))                 # standard errors via propagation of covariances in matrix form
    
    seg_wtd_ykx[,k] <- swdxm %*% bp                                   # each column of holds the seg_wtd_ykx's for one lag time
    seg_wtd_ykx_se[,k] <- sqrt(diag( swdxm %*% bb %*% t(swdxm) ))     # standard errors via propagation of covariances in matrix form
    
    avg_ykx[,k] <- adxm %*% bp                                        # each column of holds the avg_ykx's for one lag time (nx of these)
    avg_ykx_se[,k] <- sqrt(diag( adxm %*% bb %*% t(adxm) ))           # standard errors via propagation of covariances in matrix form
    
    wtd_avg_ykx[,k] <- wadxm %*% bp                                   # each column of holds the wtd_avg_ykx's for one lag time (nx of these)
    wtd_avg_ykx_se[,k] <- sqrt(diag( wadxm %*% bb %*% t(wadxm) ))     # standard errors via propagation of covariances in matrix form
  }
  
  
  ykx <- t(ykx)                                                       # transpose these back so that the each row corresponds to one lag time
  ykx_se <- t(ykx_se)
  seg_wtd_ykx <- t(seg_wtd_ykx)
  seg_wtd_ykx_se <- t(seg_wtd_ykx_se)
  avg_ykx <- t(avg_ykx)
  avg_ykx_se <- t(avg_ykx_se)
  wtd_avg_ykx <- t(wtd_avg_ykx)
  wtd_avg_ykx_se <- t(wtd_avg_ykx_se)
  
  # now convert these ykx's to beta's by dividing by the knot values
  beta <- ykx / matrix(as.vector(xknots[-1, ]), byrow=TRUE, ncol=nxk*nx, nrow=nbeta)            # need to throw away the first row of xknots because these are zeroes
  beta_se <- ykx_se / matrix(as.vector(xknots[-1, ]), byrow=TRUE, ncol=nxk*nx, nrow=nbeta)
  
  # and convert seg_wtd_ykx's to beta's by dividing by seg_wtd_meanx's
  seg_wtd_IRF <- seg_wtd_ykx / matrix(as.vector(seg_wtd_meanx), byrow=TRUE, ncol=nxk*nx, nrow=nbeta)    
  seg_wtd_se <- seg_wtd_ykx_se / matrix(as.vector(seg_wtd_meanx), byrow=TRUE, ncol=nxk*nx, nrow=nbeta)
  
  # and convert avg_ykx's to *weighted average* beta's by dividing by sum of average xprimes
  wtd_avg_IRF <- avg_ykx / matrix(rowSums(avg_xprime), byrow=TRUE, ncol=nx, nrow=nbeta)    
  wtd_avg_se <- avg_ykx_se / matrix(rowSums(avg_xprime), byrow=TRUE, ncol=nx, nrow=nbeta)
  
  # and convert wtd_avg_avg_ykx's to *square-weighted* beta's by dividing by sum of weighted xprimes
  wtd2_avg_IRF <- wtd_avg_ykx / matrix(rowSums(wtd_xprime), byrow=TRUE, ncol=nx, nrow=nbeta)    
  wtd2_avg_se <- wtd_avg_ykx_se / matrix(rowSums(wtd_xprime), byrow=TRUE, ncol=nx, nrow=nbeta)
  
  
  
  # beta is indexed by [lag, nx*nxk]
  # xprime is indexed by [time, nx*nxk]
  # betaprime is indexed by [lag, nx*nxk]
  # IRF is indexed by [lag, nx*nxk]
  # avg_IRF is indexed by [lag, nx]
  # 
  # ykx is the crossproduct (across rows) of xprime[time, nxk] and betaprime[lag, nxk]
  
  
  
  xknots = xknots[-1, , drop=FALSE] # now discard first row (zeroes) from knots
  
  
  #########################
  # now create column names
  
  # if vector of xnames does not exist, create it
  if (is.null(xnames) | length(xnames)!=ncol(x)) xnames <- paste0("x", 1:nx)
  
  ####################################
  # create vector of xnames with knots
  xknames <- rep("", nx*nxk)  # create vector for column names with x's and knots
  xwnames <- rep("", nx*nxk)  # create vector for column names with x's and mean x's between knots
  for (i in 1:nx) for (k in 1:nxk) {
    xstr <- xnames[i]
    nch <- nchar(xstr)
    ii <- 0  # need to search for location of first "|" by brute force, because | is a special character so we can't use regexpr!
    repeat{
      ii <- ii+1
      if (substr(xstr, ii, ii)=="|") {break}
      if (ii==nch) {break}
    }
    if ((ii==nch) || (ii==1)) xknames[k+(i-1)*nxk] <- paste0(xstr, "=", sprintf("%g", xknots[k,i]))   # if there is no "|" divider or it comes at the end or beginning
    else xknames[k+(i-1)*nxk] <- paste0(substr(xstr, 1, ii-1), "=", sprintf("%g", xknots[k,i]), substr(xstr, ii, nch))
    
    # original IRFnnhs code:
    # if ((ii==nch) || (ii==1)) xwnames[k+(i-1)*nxk] <- paste0(xstr, "=", sprintf("%g", seg_wtd_meanx[k,i]))   # if there is no "|" divider or it comes at the end or beginning
    # else xwnames[k+(i-1)*nxk] <- paste0(substr(xstr, 1, ii-1), "=", sprintf("%g", seg_wtd_meanx[k,i]), substr(xstr, ii, nch))
    
    # modified code to label columns with interval between xknots rather than weighted average
    if (k==1) {
      rangestr <- paste0("<", sprintf("%g", xknots[k,i]))                   # rangestr is a string for the range between pairs of xknots
    } else {
      if (k==nxk) rangestr <- paste0(">", sprintf("%g", xknots[(k-1),i]))
      else rangestr <- paste0("=", sprintf("%g", xknots[(k-1),i]), "-", sprintf("%g", xknots[k,i]))
    }
    if ((ii==nch) || (ii==1)) xwnames[k+(i-1)*nxk] <- paste0(xstr, rangestr)   # if there is no "|" divider or it comes at the end or beginning
    else xwnames[k+(i-1)*nxk] <- paste0(substr(xstr, 1, ii-1), rangestr, substr(xstr, ii, nch))
  }
  
  colnames(beta) <- paste0("IRF_", xknames)
  colnames(beta_se) <- paste0("se_", xknames)
  colnames(ykx) <- paste0("ykx_", xknames)
  colnames(ykx_se) <- paste0("ykx_se_", xknames)
  colnames(seg_wtd_IRF) <- paste0("seg_wtd_IRF_", xwnames)
  colnames(seg_wtd_se) <- paste0("seg_wtd_se_", xwnames)
  colnames(seg_wtd_ykx) <- paste0("seg_wtd_ykx_", xwnames)
  colnames(seg_wtd_ykx_se) <- paste0("seg_wtd_ykx_se_", xwnames)
  colnames(avg_ykx) <- paste0("avg_ykx_", xnames)
  colnames(avg_ykx_se) <- paste0("avg_ykx_se_", xnames)
  colnames(wtd_avg_IRF) <- paste0("wtd_avg_IRF_", xnames)
  colnames(wtd_avg_se) <- paste0("wtd_avg_se_", xnames)
  colnames(wtd2_avg_IRF) <- paste0("wtd_avg_IRF_", xnames)
  colnames(wtd2_avg_se) <- paste0("wtd_avg_se_", xnames)
  colnames(wtd_avg_ykx) <- paste0("wtd_avg_ykx_", xnames)
  colnames(wtd_avg_ykx_se) <- paste0("wtd_avg_ykx_se_", xnames)
  
  
  
  
  
  
  # return results
  return(
    list(
      lags = zz$lags ,    # lags (in number of time steps)
      nxk = rep(nxk, nx) , # number of nonlinearity knots (vector of values for each x)
      xknots = xknots ,     # nonlinearity knots, with first row (zeroes) removed
      IRF = beta ,           # nonlinear impulse response function (beta) evaluated at each knot (except zero): matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
      se = beta_se ,          # standard errors of IRF coefficients (matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
      ykx = ykx ,              # nonlinear impulse response, expressed as contribution to y from x, evaluated at each knot (except zero): matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
      ykx_se = ykx_se ,         # standard errors of ykx (matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
      wtd_avg_IRF = wtd_avg_IRF ,# weighted average of IRFs 
      wtd_avg_se = wtd_avg_se ,   # standard error of weighted average of IRFs
      avg_ykx = avg_ykx ,          # time-averaged ykx
      avg_ykx_se = avg_ykx_se ,     # standard error of time-averaged ykx
      wtd2_avg_IRF = wtd2_avg_IRF ,  # x^2-weighted average of IRF
      wtd2_avg_se = wtd2_avg_se ,     # standard error of x^2-weighted average of IRF
      wtd_avg_ykx = wtd_avg_ykx ,      # averaged ykx weighted by input
      wtd_avg_ykx_se = wtd_avg_ykx_se , # standard error of weighted average ykx
      seg_wtd_meanx = seg_wtd_meanx ,    # weighted mean of x between each pair of knots
      seg_wtd_IRF = seg_wtd_IRF ,         # weighted mean of IRF between each pair of knots
      seg_wtd_se = seg_wtd_se ,            # standard error
      seg_wtd_ykx = seg_wtd_ykx ,           # weighted mean ykx between each pair of knots
      seg_wtd_ykx_se = seg_wtd_ykx_se ,      # standard error
      Kbb = zz$Kbb ,            # stack of covariance matrices for each x and each knot (except zero) (array of nx*nxk m+1 by m+1 matrices)
      n = zz$n ,                 # length of original x and y series
      n.eff = zz$n.eff ,          # effective sample size, accounting for uneven weighting
      n.nz = zz$n.nz ,             # number of nonzero values with nonzero weight at each lag of each x variable
      h = zz$h ,                    # order of AR correction that was applied
      phi = zz$phi ,                 # fitted AR coefficients (vector of length h)
      resid = zz$e ,                  # residuals
      resid.acf = zz$resid.acf ,       # autocorrelation function of residuals
      resid.pacf = zz$resid.pacf ,      # partial autocorrelation function of residuals
      ycomp = zz$ycomp                   # data table comparing measured and fitted y time series
      
      
    )
  )
  
  
}  # end nonlinIRF

#///////////////////
# END of nonlinIRF
#///////////////////














#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
####     This is the end of the functions from IRFnnhs.R  (as modified for use here)         ####
#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
















#/////////////////////////////////////////////////////////////////////////
#### qfit: quadratic fit using pooled standard errors of input points ####
#/////////////////////////////////////////////////////////////////////////

qfit <- function( x, y, se ) {      # quadratic fitter for the peakstats routine, below
  
  if ((length(x) != length(y)) | (length(se) != length(y))) stop ("error in qfit: input lengths don't match")
  
  y[is.na(x)] <- NA      # remove any NAs
  y[is.na(se)] <- NA           #IF TOO MANY OF THE SE'S ARE NA, THEN ALL OF THE Y'S AND X'S VANISH, AND THEN THE SOLUTION FAILS!
  x <- x[!is.na(y)]
  se <- se[!is.na(y)]
  y <- y[!is.na(y)]
  n <- length(y)
  
  if (n<2) {        # not enough valid points
    return(list(
      a = NA,       # quadratic coefficient
      b = NA,       # linear coefficient
      c = NA,       # constant coefficient
      a.se = NA,
      b.se = NA,
      c.se = NA,
      covar = matrix(NA, nrow=3, ncol=3)          # covariance matrix of coefficients
    ))
  } else {
    if ((n==2) | (abs(cor(x,y))>0.999)) {  # linear relationship
      xx <- cbind(x,1)    # just do linear fit
      C <- crossprod(xx)/n
      xy <- crossprod(xx, y)/n
      
      beta.hat <- solve(C, xy)
      e <- y - (xx %*% beta.hat)  # residuals
      
      if (n>2) s2e <- sum(e*e)/(n-2) + mean(se*se) 
      else s2e <-  mean(se*se) # here we use the pooled standard errors to estimate the residual variance
      # this protects against artificially small uncertainty estimates when points "just happen" to line up in a quadratic
      
      if (n > 5) {
        ser_corr <- as.numeric(cor(e, dplyr::lag(e), use="complete.obs"))
        if (ser_corr>0) n <- n*(1-ser_corr)/(1+ser_corr)       # adjust for serial correlation in residuals
      }
      
      covar <- (s2e/n)*solve(C)
      beta.se <- sqrt(diag(covar))
      
      beta.hat <- c(0, beta.hat)                                       # add missing zero for quadratic term
      beta.se <- c(0, beta.se)                                         # add missing zero for quadratic term
      covar = as.matrix(cbind(c(0,0,0), rbind(c(0,0), covar)))         # add missing row and column to covar matrix
      
    } else {   # fit quadratic
      
      xx <- cbind(x*x, x, 1)
      
      
      C <- crossprod(xx)/n
      xy <- crossprod(xx, y)/n
      
      beta.hat <- solve(C, xy)
      e <- y - (xx %*% beta.hat)  # residuals
      
      if (n > 3) s2e <- sum(e*e)/(n-3) + mean(se*se)
      else s2e <- mean(se*se)     # here we use the pooled standard errors to estimate the residual variance
      # this protects against artificially small uncertainty estimates when points "just happen" to line up in a quadratic
      
      if (n > 5) {
        ser_corr <- as.numeric(cor(e, dplyr::lag(e), use="complete.obs"))
        if (ser_corr>0) n <- n*(1-ser_corr)/(1+ser_corr)       # adjust for serial correlation in residuals
      }
      
      covar <- (s2e/n)*solve(C)
      beta.se <- sqrt(diag(covar))
      
    }
    
  }
  
  return(list(
    a = beta.hat[1],       # quadratic coefficient
    b = beta.hat[2],       # linear coefficient
    c = beta.hat[3],       # constant coefficient
    a.se = beta.se[1],
    b.se = beta.se[2],
    c.se = beta.se[3],
    covar = covar          # covariance matrix of coefficients
  ))
  
  
}


#////////////
# END OF qfit
#////////////







#////////////////////////////////////////////////////////////////////////
#### linfit: linear fit using pooled standard errors of input points ####
#///////////////////////// //////////////////////////////////////////////

linfit <- function( x, y, se ) {      # linear fitter for the peakstats routine, below
  
  if ((length(x) != length(y)) | (length(se) != length(y))) stop ("error in linfit: input lengths don't match")
  
  y[is.na(x)] <- NA      # remove any NAs
  y[is.na(se)] <- NA           #IF TOO MANY OF THE SE'S ARE NA, THEN ALL OF THE Y'S AND X'S VANISH, AND THEN THE SOLUTION FAILS!
  x <- x[!is.na(y)]
  se <- se[!is.na(y)]
  y <- y[!is.na(y)]
  n <- length(y)
  
  if (n<2) {        # not enough valid points
    return(list(
      b = NA,       # linear coefficient
      c = NA,       # constant coefficient
      b.se = NA,
      c.se = NA,
      covar = matrix(NA, nrow=2, ncol=2)          # covariance matrix of coefficients
    ))
    
  } else {
    xx <- cbind(x,1)    # just do linear fit
    C <- crossprod(xx)/n
    xy <- crossprod(xx, y)/n
    
    beta.hat <- solve(C, xy)
    e <- y - (xx %*% beta.hat)  # residuals
    
    if (n>2) s2e <- sum(e*e)/(n-2) + mean(se*se) 
    else s2e <- mean(se*se) # here we use the pooled standard errors to estimate the residual variance
    # this protects against artificially small uncertainty estimates when points "just happen" to line up
    
    if (n > 5) {
      ser_corr <- as.numeric(cor(e, dplyr::lag(e), use="complete.obs"))
      if (ser_corr>0) n <- n*(1-ser_corr)/(1+ser_corr)       # adjust for serial correlation in residuals
    }
    
    covar <- (s2e/n)*solve(C)
    beta.se <- sqrt(diag(covar))
    
    
  }
  
  return(list(
    b = beta.hat[1],       # linear coefficient
    c = beta.hat[2],       # constant coefficient
    b.se = beta.se[1],
    c.se = beta.se[2],
    covar = covar          # covariance matrix of coefficients
  ))
  
  
}


#//////////////
# END OF linfit
#//////////////









#///////////////////////////////////////////////////////////////////
#### peakstats: find RRD peak, and measure its height and width ####
#///////////////////////////////////////////////////////////////////

peakstats <- function(lagtime, bb, se, Qavgd=TRUE ) {
  # takes as input:
  # lagtime         vector containing lag times
  # bb              vector containing RRD coefficients
  # se              vector of standard errors of RRD coefficients
  # Qavgd           boolean flag indicating whether Q is time-averaged over each interval (TRUE) or instantaneous at end of interval (FALSE)
  #                       This matters for peak timing, because for Qavgd=TRUE, the times associated with each lag are 0.3333, 1, 2, 3, 4... lag units
  #                       whereas for Qavgd=FALSE, the times associated with each lag are 0.5, 1.5, 2.5, 3.5 ... lag units.
  
  # returns a list as output, with the following objects
  # tpeak      time of peak (from quadratic fit)
  # tpeak_se   standard error of time of peak 
  # peakht     peak height (from quadratic fit)
  # peakht_se  standard error of peak height
  # alt_tpeak  time of peak (lag of single highest RRD value, as a sanity check)  !! Note that this is suppressed in builds after 2023.07.04
  # alt_peakht peak height (single highest RRD value, as a sanity check)          !! Note that this is suppressed in builds after 2023.07.04 
  # alt_peakht_se standard error of single highest RRD value (as a sanity check)  !! Note that this is suppressed in builds after 2023.07.04
  # width      width of peak, as full width at half maximum
  # width_se   standard error of width of peak
  # rc         runoff coefficient: integral of RRD
  # rc_se      standard error of runoff coefficient
  
  
  
  # add (0,0) to beginning of input vectors
  lagtime <- c(0, lagtime)
  bb <- c(0, bb)
  se <- c(0, se)
  
  #//////////////////////////////////////
  # first do peak height and time to peak
  
  peakbb <- max(bb)    # maximum of RRD 
  pk <- min( which( (bb==max(bb)) ) )  # lag number at peak
  
  before <- bb[1:pk]                                                 # subset of bb up to and including peak
  suppressWarnings( ff <- max(which( (before<0.8*peakbb) ) ) )       # last index before peak that is below 80% of peak
  after <- bb[pk:length(bb)]                                         # subset of bb up to and including peak
  suppressWarnings( ll <- min(which( (after<0.8*peakbb) ) ) + pk-1)  # first index after peak that is below 80% of peak
  
  if ((peakbb <=0) | is.infinite(ll) | is.infinite(ff)) {                               # if no feasible solution
    peakht <- NA
    peakht_se <- NA
    tpeak <- NA
    tpeak_se <- NA
    FWHM <- NA
    FWHM_se <- NA
  } else {
    
    # fit a quadratic (will work also with just 3 points because we include SE's of individual points in qfit)
    
    x <- lagtime[ff:ll] - lagtime[pk]  # x is"centered" at peak, so parameter uncertainties will be mostly uncorrelated and we can use Gaussian error propagation
    y <- bb[ff:ll]
    s <- se[ff:ll]
    
    fit <- qfit(x, y, s) 
    list2env(fit, envir=environment())    # unpack list from qfit
    
    # the quadratic reaches a maximum at ymax = c - b^2/4a .  
    
    peakht <- c - (b^2)/(4*a)
    peakht_se <- sqrt( c.se^2 + ((2*b/(4*a))*b.se)^2 + ((b^2)/(4*a^2)*a.se)^2 
                       - 4*b^3/(16*a^3)*covar[1,2]
                       + 2*b^2/(4*a^2)*covar[1,3]
                       - 4*b/(4*a)*covar[2,3])      # Error propagation including covariances
    
    # and the corresponding value of x is x_ymax = - b/2a .  Adding back in x2 we get
    
    tpeak <- lagtime[pk] - b/(2*a)
    tpeak_se <- sqrt( ((0.5/a)*b.se)^2 + (b/(2*a^2)*a.se)^2 - 2*b/(4*a^3)*covar[1,2] ) 
    # Error propagation including covariances (note that lagtime[pk] is just a reference value so we don't count its uncertainty -- would be double-counting)
    
    #///////////////////////////////////////////////////////////////////////////////////
    # Now we estimate the peak width, as expressed by full width at half maximum (FWHM)
    
    # first, find the point where the rising limb crosses half of the maximum
    
    
    hm <- 0.5*peakbb                                  # half of peak RRD value
    suppressWarnings( ff <- max(which( (before<0.4*peakbb) ) ) )  # last index before peak that is below 40% of peak (this must exist because first point has value of 0)
    snip <- bb[ff:pk] # 
    suppressWarnings( ll <- min(which( (snip>0.6*peakbb) ) ) + ff-1 )  # first index between ff and peak that is above 60% of peak (this might be the peak itself)
    
    
    if (ll-ff==1) {   # if we only have two points, do linear interpolation
      first_cross <- lagtime[ff] + (hm - bb[ff])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])
      dzdy1 <- (hm - bb[ll])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])^2
      dzdy2 <- (hm - bb[ff])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])^2
      first_cross_se <- sqrt( (dzdy1*se[ff])^2 + (dzdy2*se[ll])^2 + ((lagtime[ff]-lagtime[ll])/(bb[ff]-bb[ll])*peakht_se/2)^2)         # Gaussian error prop; last term propagates uncertainty in peakht (and thus in hm)
      
    } else {      # if we have 3 or more points, fit a line and find the root
      x <- lagtime[ff:ll]           # x 
      y <- bb[ff:ll] - hm           # y is relative to half-maximum, so the root of the quadratic will be where it crosses the half maximum
      s <- se[ff:ll]
      
      fit <- linfit(x, y, s) 
      list2env(fit, envir=environment())    # unpack list from linfit
      
      first_cross <- -c/b    # take the root
      
      dzdb <- c/b^2
      dzdc <- -1/b
      first_cross_se <- sqrt( (dzdb*b.se)^2 + (dzdc*c.se)^2 
                              + 2*dzdb*dzdc*covar[1,2]
                              + (1/b*peakht_se/2)^2 )      # last term propagates uncertainty in peakht
      
    }
    
    # now, do the whole thing over again to find the point where the falling limb crosses half of the maximum
    
    suppressWarnings( ll <- min(which( (after<0.4*peakbb) ) ) + pk-1 )  # first index after peak that is below 40% of peak (if this doesn't exist, then we say that FWHM is undefined)
    if (is.infinite(ll)) {
      FWHM <- NA
      FWHM_se <- NA
    } else {
      
      snip <- bb[pk:ll] # 
      suppressWarnings( ff <- max(which( (snip>0.6*peakbb) ) ) + pk-1 ) # last index between peak and ll that is above 60% of peak
      
      if (ll-ff==1) {   # if we only have two points, do linear interpolation
        last_cross <- lagtime[ff] + (hm - bb[ff])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])
        dzdy1 <- (hm - bb[ll])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])^2
        dzdy2 <- (hm - bb[ff])*(lagtime[ll] - lagtime[ff])/(bb[ll] - bb[ff])^2
        last_cross_se <- sqrt( (dzdy1*se[ff])^2 + (dzdy2*se[ll])^2 + ((lagtime[ff]-lagtime[ll])/(bb[ff]-bb[ll])*peakht_se/2)^2)            # Gaussian error prop; last term propagates uncertainty in peakht (and thus in hm)        
        
      } else {      # if we have 3 or more points, fit a line and find the root
        x <- lagtime[ff:ll]           # x 
        y <- bb[ff:ll] - hm           # y is relative to half-maximum, so the root of the quadratic will be where it crosses the half maximum
        s <- se[ff:ll]
        
        fit <- linfit(x, y, s) 
        list2env(fit, envir=environment())    # unpack list from linfit
        
        last_cross <- -c/b    # take the root
        
        dzdb <- c/b^2
        dzdc <- -1/b
        last_cross_se <- sqrt( (dzdb*b.se)^2 + (dzdc*c.se)^2 
                               + 2*dzdb*dzdc*covar[1,2]
                               + (1/b*peakht_se/2)^2 )      # last term propagates uncertainty in peakht
        
      }
      
      FWHM <- last_cross - first_cross
      FWHM_se <- sqrt( last_cross_se^2 + first_cross_se^2 )
    }
    
  }
  
  
  
  # now we estimate the runoff coefficient by integrating the RRD
  # remember that 0 has been added as a first element of lagtime, bb, and se
  m1 <- length(bb)
  rcwt <- rep(0, m1)
  rcwt[1] <- 0.5*lagtime[2]
  rcwt[m1] <- 0.5*(lagtime[m1]-lagtime[m1-1])
  for (i in 2:(m1-1)) rcwt[i] <- 0.5*(lagtime[i+1]-lagtime[i-1])
  
  rc <- sum(bb * rcwt)        # integral of linear interpolation between points

  rc_se <- sqrt(sum((se*rcwt)^2))
  
  
  if (!is.na(peakht)) {
    alt_tpeak <- lagtime[pk]
    alt_peakht <- bb[pk]
    alt_peakht_se <- se[pk]
  } else {
    alt_tpeak <- NA
    alt_peakht <- NA
    alt_peakht_se <- NA
  }
  
  
  return(
    list(
      tpeak = tpeak ,           # time of peak (from quadratic fit)
      tpeak_se = tpeak_se ,      # standard error
      peakht = peakht ,           # peak value of RRD (from quadratic fit)
      peakht_se = peakht_se ,      # standard error
      width = FWHM ,                   # full width at half maximum of RRD peak
      width_se = FWHM_se ,              # standard error
      rc = rc ,                          # integral of RRD (should approximate q/p)
      rc_se = rc_se                       # standard error of sum of RRD
    )
  ) # end return
  
}

#/////////////////
# END OF peakstats
#/////////////////








#///////////////////////////////////
#### make subsets for splitting ####
#///////////////////////////////////

make_sets <-
  function(n , np=1, wt=rep(1, n), split_params = NULL, p_label) {
    
    # defines sets according to multiple (possibly nested) criteria for splitting RRDs in the following function
    
    # takes as input:
    # n              number of time steps (n=length(p))
    # np             number of separate p time series (default=1)
    # wt             optional weight vector (must be of same length as NROW(p) or q).  This is used only to calculated weighted means for binmean.
    # split_params   a list containing all the parameters needed for splitting.  These are:
    #   crit         a list of n_crit criterion variables, each of which is a vector or matrix of same length as p and q.
    #                      If any element of crit is a matrix, it must be np columns wide, and each column will be used to split the corresponding column
    #                      of the precipitation matrix (if the precipitation matrix is more than one column wide -- in other words, if we have multiple
    #                      precipitation inputs).  Any element of crit that is a vector or a one-column matrix will be applied to all columns of the
    #                      precipitation matrix jointly.  
    #                      Note that precipitation rates should normally NOT be used as a splitting variable
    #                      (although lagged precipitation rates could potentially be).  Nonlinear dependence on precipitation rates should 
    #                      be handled instead by setting xknots in the call to ERRA.  
    #   crit_label   a vector of n_crit strings that label the criterion variables
    #   crit_lag     a vector of n_crit integers, indicating how many time steps each criterion should be lagged.  Values <1 result in no lagging.  
    #   pct_breakpts a Boolean vector of length n_crit, indicating whether each criterion's breakpoints are given 
    #                      in the units of the criteria themselves (pct_breakpts==FALSE) or in percentiles (pct_breakpts==TRUE).
    #   breakpts     a list of n_crit vectors of breakpoint values that will be used to divide the criterion variables into bins.  
    #                      For example, if the first variable is divided at values of 0.3 and 0.6, and the second variable is divided at 
    #                      percentiles of 70, 80, and 90 percent, then breakpts=list(c(0.3,0.6), c(70,80,90)) and pct_breakpts=c(FALSE, TRUE).  
    #                      Each vector within the breakpts list must be in ascending order.
    #   thresh       a vector of n_crit threshold values.  Criterion values <= threshold are ignored in setting break points according to percentile ranks 
    #                      (if pct_breakpts==TRUE; thresh has no effect if pct_breakpts==FALSE).  Thresholds can be helpful when a criterion variable has many
    #                      trivially small values that should all go in the lowest bin, with the breakpoints being set within in the range of the criterion
    #                      values that are not trivially small.  Note that criterion values below thresh will still be used in splitting the data set, 
    #                      just not in setting the values of the breakpoints.
    #   by_bin       a Boolean vector of length n_crit, indicating whether each criterion's breakpoints are set separately within each bin of the previous criteria,
    #                      or as one set of breakpoints for the whole data set.  For example, if by_bin==TRUE for criterion #3, criterion 3 is applied separately within
    #                      each bin formed by criteria #1 and #2.  That is, the bin limits for criterion 3 are are set according to the percentiles of criterion 3's values 
    #                      among cases that conform to a given bin of criterion 1 and 2 (rather than the percentiles of all cases, which are used if by_bin==FALSE).
    #                      This is useful when criteria are interdependent, such that (for example) the smallest bin of criterion 2 may have no values that also fit into
    #                      the largest bin of criterion 1.  
    #                      If pct_breakpts==FALSE for a criterion, by_bin has no effect on the binning of that criterion.
    #                      by_bin has no effect on the first criterion (there is nothing yet to nest within).
    #
    # p_label        a vector of string labels for each column of the precipitation matrix (length=np)
    
    # if split_params==NULL, no splitting is performed
    
    # returns a list as output, with the following objects
    # n_sets       number of different sets of RRDs
    # n_crit       number of different criteria
    # sets         matrix of n rows and np columns of integers indicating which set of criterion bins that time step belongs to
    # lwr          lower limit of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    # upr          upper limit of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    # binmean      average of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    # set_label    vector of np*n_sets character string labels for each set
    
    
    if (is.null(split_params)) {  # if split_params==NULL, then make one set for everything, and return
      
      set_label = rep("blank", np)
      for (j in 1:np) set_label[j] <- paste0(p_label[j], "|all")
      return(
        list(
          n_sets = 1 ,
          n_crit = 0 ,
          sets = matrix(1, nrow=n, ncol=np) ,
          lwr = matrix(0, nrow=np , ncol=1) ,
          upr = matrix(0, nrow=np, ncol=1) ,
          binmean = matrix(0, nrow=np, ncol=1) ,
          set_label = set_label 
        ) # end list
      ) # end return
    }
    
    
    # if split_params is not NULL, then we go forward here...
    
    
    list2env(split_params, env=environment())    # "unpack" split_params in the local environment of this function
    
    if (typeof(crit) != "list") stop("Error in ERRA: crit must be a list (not, e.g., a vector that has not been enclosed in list())")
    if (typeof(breakpts) != "list") stop("Error in ERRA: breakpts must be a list (not, e.g., a vector that has not been enclosed in list())")
    
    n_crit <- length(crit)                       # number of different criterion variables
    
    if (length(pct_breakpts) != n_crit) stop("Error in ERRA: pct_breakpts vector doesn't match number of criteria!")
    if (length(breakpts) != n_crit) stop("Error in ERRA: breakpts list doesn't match number of criteria!")
    if (length(thresh) != n_crit) stop("Error in ERRA: thresh vector doesn't match number of criteria!")
    if (length(by_bin) != n_crit) stop("Error in ERRA: by_bin vector doesn't match number of criteria!")
    
    for (i in 1:n_crit) {
      crit[[i]] <- as.matrix(crit[[i]])          # coerce each criterion variable to matrix
      
      if (NROW(crit[[i]]) != n)  stop("Fatal error in ERRA: a criterion vector or matrix is the wrong length")
      if (NCOL(crit[[i]]) > 1) if (NCOL(crit[[i]]) != np) stop("Fatal error in ERRA: column count of each criterion vector or matrix must equal either 1 or number of columns of p")
      
      if ( (max(breakpts[[i]])<1) && (pct_breakpts[i]==TRUE) ) breakpts[[i]] <- breakpts[[i]]*100  # if pct_breakpts is TRUE but max(breakpts)<1, assume users have mistakenly specified quantiles and recalculate as percentiles
    } 
    
    for (i in 1:n_crit) if (crit_lag[i] > 0) crit[[i]] <- dplyr::lag(crit[[i]], crit_lag[i])  # here we can just lag the criterion by whatever the specified lag is
    
    for (ii in 1:np) {  # step through each precipitation series in order
      
      ss <- 1
      for (i in 1:n_crit) ss <- ss*(1+length(breakpts[[i]]))  # multiply the number of bins for each criterion to get the total number of sets
      
      sets <- matrix(1, nrow=n, ncol=1)                      # initialize set matrix to all 1's (there is only one set for now)
      lwr <- matrix(NA, nrow=ss, ncol=n_crit)                # matrix of lower bounds for each bin of each criterion (rows are set numbers, columns are criterion indices)
      upr <- matrix(NA, nrow=ss, ncol=n_crit)                # matrix of upper bounds for each bin of each criterion (rows are set numbers, columns are criterion indices)
      n_sets <- 1                                            # initialize previous number of sets (this will be updated and at the end will be the final number of bins)
      
      # now, for each precipitation series we assign bin boundaries and create set numbers
      
      for (i in 1:n_crit) {   # step through each criterion in order
        
        old_sets <- sets                          # copy the old set matrix, which will be updated
        
        cr <- crit[[i]]                           # extract the current criterion matrix from the list "crit" 
        if (NCOL(cr)>1) cr <- cr[,ii]             # if there are multiple columns in this criterion matrix, extract the one that corresponds to this precipitation series
        
        bp <- breakpts[[i]]                        # extract the current breakpoint vector from the list "breakpts"
        th <- thresh[[i]]                         # extract the current threshold value from the list "thresh"
        n_bins <- 1+length(bp)                    # number of bins for this criterion (= number of breakpoints + 1)
        
        if (i>1) {                   # this copies the prior bin boundaries from the previous criteria
          old_lwr <- lwr
          old_upr <- upr
          for (j in 1:(n_sets*n_bins)) {          # note that this is the OLD n_sets times the NEW n_bins... which is what we want
            jj <- ceiling((j-0.5)/n_bins)
            lwr[j,] <- old_lwr[jj,]               # this copies the old bin boundaries to the new bins, with the proper interleaving
            upr[j,] <- old_upr[jj,]
          }
        }
        
        
        if (length(bp)>1)  # check that breakpoints are sorted (bp is the set of breakpoints for this particular criterion)
          if( min(diff(bp), na.rm=TRUE) <=0 ) stop("Fatal error in ERRA: breakpoints must be unique and sorted in ascending order")
        
        
        if (pct_breakpts[[i]]==FALSE) {   # check that breakpoint values are not out of range
          if (by_bin[[i]]) stop("Fatal error in ERRA: by_bin breakpoints should be fixed by percentiles, not by values")
          if ( ( min(bp) <= min(cr[(cr>th)], na.rm=TRUE) ) | (max(bp) >= max(cr, na.rm=TRUE)) )
            stop("Fatal error in ERRA: one or more breakpoints are outside range of criterion variable (and/or threshold)")
        } else {
          qbp <- bp                              # rename breakpoint vector as qbp (quantile breakpoints) -- this is because we will need to recycle the breakpoint vector
        }
        
        
        if (by_bin[[i]]) {                        # if breakpoint quantiles are nested into (i.e., applied separately to) each set of previous breakpoint bins
          for (j in 1:n_sets) {                   # then cycle through the extant sets from previous criteria
            
            crss <- cr[((cr>th) & (old_sets==j))]               # criterion subset (above threshold, only values within the current set)
            bp <- quantile(crss, probs=qbp/100.0, na.rm=TRUE, type=5)    # convert quantile breakpoints to criterion values
            crss <- cr[(old_sets==j)]          # different criterion subset (NO threshold, only values within the current set)
            ll <- c(min(crss, na.rm=TRUE), bp)          # vector of lower bounds of each bin
            uu <- c(bp, max(crss, na.rm=TRUE)+1e-6 )    # vector of upper bounds of each bin
            
            for (k in 1:n_bins) sets[((old_sets==j) & (cr>=ll[k]) & (cr<uu[k]))] <- (j-1)*n_bins + k    # assign new set number to each time step now (greater than OR EQUAL TO the lower limit, and LESS THAN the upper limit)
            
            if (j==1) ll2 <- ll else ll2 <- c(ll2,ll)   # vector of lower bounds of each bin, concatenated for each (old) set
            if (j==1) uu2 <- uu else uu2 <- c(uu2,uu)   # vector of upper bounds of each bin, concatenated for each (old) set
            
          } # next j
          
          lwr[,i] <- ll2  # save lower bounds to matrix
          upr[,i] <- uu2  # save upper bounds to matrix
          
          
        } else {
          
          if (pct_breakpts[[i]]==TRUE) bp <- quantile(cr[(cr>th)], probs=qbp/100.0, na.rm=TRUE, type=5)  # convert quantiles to criterion values
          ll <- c(min(cr, na.rm=TRUE), bp)          # vector of lower bounds of each bin
          uu <- c(bp, max(cr, na.rm=TRUE)+1e-6 )    # vector of upper bounds of each bin
          
          for (k in 1:n_bins) sets[((cr>=ll[k]) & (cr<uu[k]))] <- k      # assign new set number to each time step now
          sets <- (old_sets-1)*n_bins + sets                             # merge old and new set numbers
          
          lwr[,i] <- ll  # save lower bounds to matrix (with recycling)
          upr[,i] <- uu  # save upper bounds to matrix (with recycling)
          
          
        } # end if/else by_bin
        
        n_sets <- n_sets*n_bins                                      # update n_sets
        
        sets[is.na(cr)] <- NA                # just in case: missing criterion values at any time step make sets=NA
        
      }  # next criterion
      
      if (ii==1) {                  # if this is the first precipitation series, then start the multi-precipitation matrices llwr, uupr, and ssets
        llwr <- lwr
        uupr <- upr
        ssets <- sets
      } else {                      # otherwise we update those matrices
        llwr <- rbind(llwr, lwr)
        uupr <- rbind(uupr, upr)
        ssets <- cbind(ssets, sets)
      }
      
    }  # next precipitation series
    
    
    # now that all the precip time series are done, copy uupr, llwr, and ssets back to upr, lwr, and sets
    
    lwr <- llwr
    upr <- uupr
    sets <- as.matrix(ssets)
    
    binmean <- matrix(NA, nrow=np*n_sets, ncol=n_crit)            # matrix of bin means for each bin of each criterion (rows are set numbers, columns are criterion indices)
    
    wt <- matrix(wt, ncol=np)                                     # recast wt as a matrix (so that we are consistent if weight is different for different precip time series)
    
    wt[is.na(wt)] <- 0                                            # replace any missing values of wt with zeroes to avoid problems in the weighted means below
    
    # calculate bin means
    for (ii in 1:np)       # loop through precip sources
      for (j in 1:n_sets)    # loop through sets
        for (i in 1:n_crit) {  # loop through criteria
          if (NCOL(crit[[i]])==1)
            binmean[((ii-1)*n_sets+j),i] <- weighted.mean(crit[[i]][sets[,ii]==j, 1], wt[sets[,ii]==j,ii], na.rm=TRUE)        # note that binmeans for each criterion will vary with bins of other criteria
          else binmean[((ii-1)*n_sets+j),i] <- weighted.mean(crit[[i]][sets[,ii]==j, ii], wt[sets[,ii]==j,ii], na.rm=TRUE)    # note that binmeans for each criterion will vary with bins of other criteria
        }
    
    # make set labels
    set_label <- rep("blank", np*n_sets)
    for (ii in 1:np)
      for (j in 1:n_sets) {
        lbl <- p_label[ii]
        # this labels each column by the bin mean (or means) of the criterion (or criteria)
        #        for (i in 1:n_crit) lbl <- paste0(lbl, "|", crit_label[i], format(binmean[((ii-1)*n_sets+j),i], digits=3), collapse=NULL)
        # this labels each column by the min-max range of (or ranges) of the criterion (or criteria)
        for (i in 1:n_crit) lbl <- paste0(lbl, "|", crit_label[i], format(lwr[((ii-1)*n_sets+j),i], digits=3), "-", format(upr[((ii-1)*n_sets+j),i], digits=3), collapse=NULL)
        set_label[(ii-1)*n_sets+j] <- lbl
      }
    
    if (n_sets>1) {
      colnames(lwr) <- paste0("lwr_", split_params$crit_label)
      colnames(upr) <- paste0("upr_", split_params$crit_label)
      colnames(binmean) <- paste0("mean_", split_params$crit_label)
    }
    
    
    return(
      list(
        n_sets = n_sets ,
        n_crit = n_crit ,
        sets = sets ,
        lwr = lwr ,
        upr = upr ,
        binmean = binmean ,
        set_label = set_label
      ) # end list
    ) # end return
    
    
  } # end make_sets

#//////////////////////////////////
# END OF make subsets for splitting
#//////////////////////////////////











#//////////////////////////////
#### aggregate input data ####
#//////////////////////////////

agg_data <- function(x , agg) {
  
  
  # takes as input:
  # x              vector or matrix of numeric or logical values, or a list of vectors or matrices
  # agg            aggregation factor for time steps.  Must be integer >= 1.  
  
  # returns x, aggregated into averages of agg adjacent time steps 
  
  if ((agg != round(agg)) | (agg < 1)) stop("fatal error in agg_data: agg is noninteger or <1")
  
  fv <- rep(1/agg, agg)    # filter vector that will average x over lags 0-(agg-1)
  
  if(!is.null(x)) {
    if (is.list(x)) for (i in 1:length(x)) {   # if x is a list we need to work through each element individually
      
      x[[i]] <- as.matrix(x[[i]])
      n <- NROW(x[[i]])
      d <- (1:n) %% agg           # d is row number modulo agg
      xf <- as.matrix(stats::filter(as.numeric(x[[i]]), fv, sides=1))  # average the last "agg" values
      if (is.logical(x[[i]])) x[[i]] <- as.logical(round(xf[d==0, ]))  # aggregate logical values using majority rule          
      else x[[i]] <- xf[d==0, ]  # aggregate each criterion variable (by rows)
      
    } # for loop
    
    else {                        # if x is not a list, then life is simple:
      
      x <- as.matrix(x)
      n <- NROW(x)
      d <- (1:n) %% agg           # d is row number modulo agg
      xf <- as.matrix(stats::filter(x, fv, sides=1))  # average the last "agg" values
      if (is.logical(x)) x <- as.logical(round(xf[d==0, ]))  # aggregate logical values using majority rule          
      else x <- xf[d==0, ]  # aggregate each criterion variable (by rows)
      
    }     # end else
    
  }       # if x is NULL, then just bypass all of this and return x
  
  
  return(x)
  
}

#////////////////////////////
# END OF aggregate input data
#////////////////////////////








#////////////////////
#### set lagtime ####
#////////////////////

set_lagtime <- function(lags , 
                        dt = 1, 
                        Qavgd = TRUE,
                        agg = 1)
{
  # takes as input:
  # lags           vector of lags (in number of *aggregated* time steps)
  # dt             time step in dimensional terms (before aggregation)
  # Qavgd          boolean flag for whether discharge is time-averaged (Qavgd==TRUE), or instantaneous at end of time step (Qavgd==FALSE) 
  # agg            aggregation factor for time steps.  Must be integer >= 1.
  
  
  # returns a list as output, with the following objects:
  # lagtime        a vector of m+1 lag times corresponding to the aggregated time steps (and accounting for whether Qavgd is true or false)
  #                       these are in dimensional time (depending on dt)
  # dt             time step in dimensional terms (after aggregation)
  # first_mult     a multiplication factor for the first lag, accounting for the fraction of p that cannot affect q (because it falls after q, but within the aggregated time step)
  #                      this will be affected by aggregation, and by whether q is time-averaged or not
  
  if ((agg != round(agg)) | (agg < 1)) stop("fatal error in set_lagtime: agg is noninteger or <1")
  
  dt <- agg * dt    # new dt is old dt, aggregated
  
  lagtime <- lags   # start with simple sequence of numbers
  
  # lagtimes are average time differences between a drop of water falling anytime during one time step and its effects being felt
  # in discharge (either at the end of each time step if Qavgd is FALSE, or anytime during each time step if Qavgd is TRUE)
  
  if (Qavgd==FALSE) {                 # if q is not time-averaged
    first_mult <- 2*agg/(agg+1)       # multiplier for first point
    lagtime <- lagtime + 1/(2*agg)    # lagtime (except for first time step, which is different and will be handled next)
    lagtime[1] <- (2*agg + 1)/(6*agg) # lag in first time step
  } else {                            # if q IS time-averaged
    lagtime[1] <- 1/3                 # lag in first time step (others are unchanged: 1, 2, 3, etc.)
    first_mult <- 2.0                 # multiplier for first point
  }
  
  
  return(list(
    lagtime = lagtime * dt ,                # lagtime in dimensional time (instead of number of time steps)
    first_mult = first_mult ,
    dt = dt
  ))
  
}

#////////////////
# END OF set lagtime
#////////////////

























































#//////////////////////////////////////////////////////////////////////////////////////////////////////////
#### ERRA: Ensemble Rainfall-Runoff Analysis including nonlinear dependence on precipitation intensity ####
#//////////////////////////////////////////////////////////////////////////////////////////////////////////

ERRA <-
  function(p ,
           q ,
           wt = rep(1, NROW(as.matrix(p))) ,
           m = 60 ,
           nk = 0 ,
           nu = 0.0 ,
           fq = 0 ,
           p_label = NULL ,
           split_params = NULL ,
           xknots = NULL ,
           xknot_type = NULL ,
           show_top_xknot = FALSE ,
           Qfilter = rep(1, NROW(as.matrix(p))) ,
           Qavgd = TRUE ,
           dt = 1 ,
           agg = 1 ,
           h = NULL ,
           ARprob = 0.05 ,
           ARlim = 0.2 ,
           max.AR = 6 ,
           complete.RRD = FALSE ,
           verbose = TRUE ,
           robust = FALSE ,
           max.chunk = 1e8 ,
           low.data.warn = 16 ,
           low.data.fail = 8 ,
           ser.corr.warn = 0.99)

  {
    
    #################
    # INPUTS TO ERRA
    
    
    # ERRA takes as input:
    # p               vector of precipitation rates (or a matrix or data frame, if there is more than one precipitation driver), evenly spaced in time
    #                    if p is a matrix or data frame, p is treated as ncol(p) simultaneous time series of inputs to the catchment
    #                    if p is supplied as a time series, it will be coerced to a vector or matrix, as appropriate
    # q               vector of discharge (in same units, and over the same time intervals, as p)
    #                    if q is supplied as a time series, it will be coerced to a vector
    # wt              optional vector of point weights
    # m               maximum lag for RRD, in number of time steps (number of lag steps will be m+1 to include lag of zero)
    # nk              number of knots in a piecewise linear broken stick representation of runoff response (RRD and NRF) as a function of lag.
    #                    Must be an integer greater than 2 and less than or equal to m+1, or must be zero (the default).  
    #                    If nk>2 and nk<=m+1, nk knots will be created at lags of 0, 1, and a geometric progression 
    #                    (or as close to geometric as it can be, given that the lags are integers) between lags 1 and m.
    #                    If nk<=2, nk>m+1, or nk==0 (the default), the broken-stick approach is not used, and instead 
    #                    the IRF is evaluated for m+1 lags between 0 and m.
    #                    Note that analogous broken-stick representations can also be used to estimate how runoff response changes as a function of 
    #                    precipitation intensity.  The knots for that broken-stick model are called "xknots" (see below) to distinguish them
    #                    from the knots defined here.
    # nu              fractional weight for Tikhonov-Phillips regularization (0 = un-regularized regression)
    # fq              filter quantile, for which values of 0<=fq<1 are valid; anything else is a fatal error.  If fq>0, a running quantile filter
    #                    is calculated and subtracted from q, to remove drift or seasonal patterns that could otherwise generate artifacts
    #                    in the RRD.  The running quantile filter is the fq'th quantile of a moving window of 4m+1 time steps, centered around the
    #                    time in question.  If fq==0.5, the fast running median algorithm is used; otherwise the somewhat slower moving quantile 
    #                    algorithm is used.  If fq==0, no filtering is performed.
    # p_label         an optional vector of string labels for the columns of the p matrix.  For example, if p has two columns corresponding to rain
    #                    and snow, and p_label = c("rain", "snow"), then outputs will also have those labels to identify responses to those two variables.
    #                    If p_label is not NULL, it must have as many entries as there are columns in p.  If p_label is null or if it has the wrong number
    #                    of entries, then any column names of the p matrix will be used instead.  If those column names do not exist, then values of "p1", "p2" etc. 
    #                    will be assigned (or simply "p", if p is a vector or single-column matrix).
    
    # split_params    a list containing all the parameters needed for splitting by precipitation time steps.  These are:
    #   crit          a list of n_crit criterion variables, each of which is a vector of same length as p and q
    #                       Note that precipitation itself should NOT be used as a splitting variable (although lagged values of precipitation could potentially be used).
    #                       To estimate nonlinear responses to precipitation intensity, set xknots accordingly.
    #   crit_label    a vector of n_crit strings that label the criterion variables
    #   crit_lag      a vector of n_crit integers, indicating how many time steps each criterion should be lagged.  Values <1 result in no lagging.  
    #   pct_breakpts  a boolean vector of length n_crit, indicating whether each criterion's breakpoints are given
    #                       in the units of the criteria themselves (pct_breakpts==FALSE) or in percentiles (pct_breakpts==TRUE)
    #   breakpts      a list of n_crit vectors of breakpoint values that will be used to divide the criterion variables into bins.  Each vector must be in ascending order.
    #                       for example, if the first variable is divided at values of 0.3 and 0.6, and the second variable is divided at
    #                       percentiles of 70, 80, and 90 percent, then breakpts=list(c(0.3,0.6), c(70,80,90)) and pct_breakpts=c(FALSE, TRUE).   If pct_breakpts is TRUE but the maximum
    #                       specified breakpoint is <1, ERRA will assume that quantiles (fractions of 1) have been specified instead of percentiles, and create the corresponding percentiles instead.
    #   thresh        a vector of n_crit threshold values.  Criterion values <= thresh are ignored in setting break points according to percentile ranks 
    #                       (that's if pct_breakpts==TRUE. thresh has no effect if pct_breakpts==FALSE).  Thresholds can be helpful when a criterion variable has many
    #                       trivially small values that should all go in the lowest bin, with the breakpoints being set within in the range of the criterion
    #                       values that are not trivially small.  Note that criterion values below thresh will still be used in splitting the data set, 
    #                       just not in setting the values of the breakpoints.
    #   by_bin        a boolean vector of length n_crit, indicating whether each criterion's breakpoints are set separately within each bin of the previous criteria,
    #                       or as one set of breakpoints for the whole data set.  For example, if by_bin==TRUE, criterion3 is applied separately within
    #                       each bin formed by criteria 1 and 2.  That is, the bin limits for criterion3 are are set according to the percentiles of criterion3 values
    #                       among cases that conform to a given bin of criterion1 and 2 (rather than the percentiles of all cases, which are used if by_bin==FALSE).
    #                       This is useful when criteria are interdependent, such that (for example) the smallest bin of criterion1 may have no values that also fit into
    #                       the largest bin of criterion3.  If pct_breakpts==FALSE for a criterion, by_bin has no effect on the binning of that criterion.
    
    # if split_params==NULL, no splitting is performed
    
    # xknots         a vector or matrix of knots for the piecewise linear approximation of the RRD's nonlinear dependence on p.
    #                       xknots can be specified as fixed values or as percentiles of the p distribution (depending on the xknot_type flag -- see below).
    #                                  For xknot_type=="percentiles", "cumsum" or "sqsum", for which xknots should be specified as percentiles,
    #                                  if max(xknots) is less than 1, ERRA will assume that the percentiles have been specified as quantiles (fractions) instead,
    #                                  and will convert them to percentiles.
    #                                  Values of p=0 are ignored when percentiles are later converted to fixed values (but note that very small nonzero p values will be included).
    #                       If xknots is a matrix, it must have np*n_sets columns, and each column of xknots will be applied to the corresponding (split) columns of p.  
    #                       Unless xknot_type=="even", the xknots vector (or each column of the xknots matrix) must be sorted into strictly ascending order, with no duplicate values.
    #                       If xknots is a vector or single-column matrix and p is a multi-column matrix, the same xknots will be applied to each column of p.
    #                       Xknots must be between (and not include) the minimum and maximum values of the corresponding column of p (including any splits).  
    #                       Each column of xknots must have the same number of knots, although the xknot values themselves may differ.
    #                       If xknots == NULL, potential nonlinear responses to precipitation intensity are ignored and a single RRD is estimated for each 
    #                                 column of p (and, potentially, each subset created by split_params).
    #                       If xknots != NULL, separate RRD's are estimated for each specified knot point (and also the maximum value) of each column of p 
    #                                 (and, potentially, each subset created by split_params).
    #                       If xknots != NULL and xknot_type=="even", then xknots must consist of two (and only two) positive integers, specifying the number
    #                                 of xknots and the minimum number of data points between each pair of xknots (see xknot_type=="even", below)
    #
    # xknot_type      a flag indicating how the nonlinearity knots should be used.  This can be abbreviated as the first letter of the string values below.
    #                       If xknot_type=="values", xknots are expressed as values of p.  If xknots is a vector, the same xknot values will be applied across
    #                                 all precipitation columns (including any precipitation subsets).
    #                       If xknot_type=="percentiles", xknots are expressed as percentiles of the p distribution.  If xknots is a vector, the same xknot percentiles
    #                                 (yielding different numerical values) will be applied to each precipitation column (including any precipitation subsets).
    #                       If xknot_type=="cumsum", xknots are expressed as percentiles of the cumulative sum of p (so if xknots=80, for example, 
    #                                 the knot will be the value of p for which all smaller p's sum to 80 percent of the sum of all p's).  
    #                                 Thus these xknots will delimit fractions of the total input p.  If xknots is a vector, the same cumulative sum percentiles
    #                                 (yielding different numerical values) will be applied to each precipitation column (including any precipitation subsets).
    #                       If xknot_type=="sqsum", xknots are calculated as percentiles of the cumulative sum of squares of p (so if xknots=20, for example, 
    #                                 the knot will be the value of p for which all smaller p's, squared and summed, add up to 20 percent of 
    #                                 the sum of squared p's).  Thus these xknots will delimit fractions of the total sum of squared inputs p^2.
    #                                 These will roughly approximate the corresponding fractions of the total leverage in the data set, 
    #                                 if the distribution of p's is strongly skewed with a peak near zero and a long right tail.  If xknots is a vector, 
    #                                 the same percentiles of the cumulative sum of squares (yielding different numerical values) will be applied to 
    #                                 each precipitation column (including any precipitation subsets).
    #                       If xknot_type=="even", then xknots are spaced as evenly as possible across the range of p.  If xknot_type=="even", then
    #                                 xknots must be a vector of length 2, with both values being positive integers (anything else is a fatal error).  
    #                                 The first integer indicates the number of xknots to be used (not including the uppermost xknot at max(p) or the lowermost xknot at p=0).
    #                                 The second integer indicates the minimum number of valid points in each interval between xknots (which may require a somewhat
    #                                 uneven spacing of xknots).  This second integer should be set large enough that the slope of the response (beta_prime)
    #                                 can be adequately constrained in each interval between knots, but kept small enough that the range of p spanned by
    #                                 any interval is not too wide (particularly at the upper tail of the distribution, where a wide range of p is spanned
    #                                 by relatively few data points.  If xknots is a vector, the same parameters will be used to construct separate 
    #                                 sets of xknot values for each precipitation column (including any precipitation subsets).
    #
    # show_top_xknot  boolean flag for whether the highest xknot (at the highest precipitation value in the input data) should be reported as output or not.
    #                       Default is FALSE, because values at the highest xknot tend to be unreliable, since they are controlled by correlations
    #                       between discharge and relatively few, highly skewed precipitation values
    #                       
    #                       xknots should be chosen so that (these are not checked):
    #                       (a) there are enough points in each interval between xknots, and sufficient variability in their values, to define the dependence of q on p,
    #                       (b) intervals between individual pairs of xknots do not span major changes in slope in the nonlinear relationship between q and p
    #  
    
    
    # Qfilter         optional boolean vector (1 and 0, T and F, or TRUE and FALSE) indicating whether individual discharge time steps should be included in the analysis.
    #                          This vector allows one to analyze subsets filtered according to discharge time.  To analyze multiple subsets, change Qfilter and re-run this function.
    # Qavgd           set this flag TRUE if q represents the average discharge over each sampling interval.  Set it FALSE if q represents an instantaneous value at the end of each sampling interval.
    #                          This affects the lag time that corresponds to each lag interval.  It also affects the RRD and NRF values at lag zero (same-time-step response).
    # dt              optional time step length, in whatever time units the user wants.  If step length is not supplied, then RRD returns the runoff response
    #                          distribution in fractions per time step rather than per unit time (i.e., the default value of dt is 1).
    # agg             integer aggregation factor for time steps.  If agg>1, p, and q, and splitting criteria will be combined by averaging in sets of agg time steps 
    #                          (e.g., agg=24 converts hourly data to daily averages).  Non-integer values of agg will be rounded up.  Values of agg<1 will be converted to 1 (no aggregation).
    # h               integer order of autoregressive correction (non-negative integer).  Higher-order AR corrections can, by the duality principle,
    #                          be used to correct for moving average (MA) noise as well.  If h==0, no correction is applied.  
    #                          If h==NULL, the order of autoregressive correction will be determined iteratively, as described in nonlinIRF script.
    # ARprob          significance threshold in automated AR order selection (see IRF script and K2022 for details)
    # ARlim           threshold value of residual correlation coefficients in automated AR order selection (see IRF script and K2022 for details)
    # max.AR          maximum order of AR correction that will be accepted in automatic AR order selection (see IRF script and K2022 for details)
    # complete.RRD    flag for whether the number of lagtimes will be *assumed* to be sufficient to hold the complete RRD (TRUE), meaning that any RRD
    #                          coefficients at longer lagtimes are trivially small, or whether this cannot be assumed (FALSE).  Complete.RRD=TRUE will yield
    #                          smaller, and more accurately estimated, standard errors if the real-world RRD actually does converge to zero before 
    #                          the maximum lag is reached.  But if this is not the case, complete.RRD=TRUE will force the RRD to artifactually
    #                          converge toward zero at the longest lag (with artificially small standard errors).  Complete.RRD=TRUE should thus
    #                          be invoked with caution.  
    # verbose         controls whether intermediate progress reports are printed (TRUE) or suppressed (FALSE)
    # robust          flag controlling whether robust estimation by Iteratively Reweighted Least Squares (IRLS) will be used.  A special tweak is used to prevent the solution from 
    #                          collapsing if more than half of the values of the response variable (i.e., discharge) are the same (for example, in ephemeral streams with frequent
    #                          discharges of zero).  Instead of scaling the weighting function by the median absolute residual, we scale it by the (0.5 + 0.5*(# identical y's)/(# total y's))
    #                          quantile of the absolute residuals, where "# identical y's" means the largest number of y's with the same identical value.  This defaults to the median when
    #                          we have no repeated y's, but also gives reasonable robust estimates when we have large numbers of y's with the same value.
    # max.chunk       maximum size, in bytes, of the largest piece of the design matrix (the matrix of p and its lags) that will be created
    #                          at one time.  Design matrices that would be larger than max.chunk will be created and processed
    #                          in separate "chunks" to avoid triggering memory paging, which could substantially increase runtime.  
    #                          Keeping max.chunk relatively small (order 1e8 or 1e9) incurs only a small performance penalty, except in the case of robust estimation,
    #                          where the need to iterate means that the solution will be faster (by about a factor of 2 or so) *if* one can keep the whole design matrix
    #                          in one chunk without triggering memory paging.
    # low.data.fail   failure threshold for minimum count of nonzero p values in any precipitation subset at any lag.  
    #                          A hard stop is triggered if min(n.nz) (see below for explanation of n.nz) is smaller than this.
    # low.data.warn   warning threshold for minimum count of nonzero p values in any precipitation subset at any lag.  
    #                          If min(n.nz) (see below for explanation of n.nz) is smaller than this, a warning is issued but results will be returned.
    # ser.corr.warn   warning threshold for lag-1 serial correlation of residuals.  When this is exceeded, a warning is issued suggesting that time steps should be aggregated.
    #                          Note that warning will only be issued if h is 0 or NULL; for other values of h this test would not have the same relevance.
    #                          Warning is also suppressed when using broken-stick linear interpolation over lags (i.e., if nk>2).
    
    #                 q, p, and wt can contain missing values.  Missing values of q create 1+h missing rows in the regression matrix.
    #                 Missing values of p create m+1 missing rows (one for each lag) in the regression matrix.
    
    
    # Here is an example of a set of splitting criteria for split_params:
    # 
    # split_crit <- list(crit = list(T, q) ,                          # here we designate two variables we will use as splitting criteria: temperature and lagged discharge.
    #                                                                 # that means, we divide precipitation time steps into groups according to temperature
    #                                                                 # and then according to lagged discharge (see crit_lag) as a proxy measure of catchment wetness.
    #                                                                 # Both variables must be present in the calling environment, and both must be of the same length as q.
    #
    #             crit_label = c("T", "lagQ") ,                       # these are labels so that we can later tell which group is which 
    #
    #             crit_lag = c(0, 1) ,                                # this says we will lag the second criterion variable (q) by one time step, and won't lag the first criterion variable (T)
    #
    #             pct_breakpts = c(FALSE, TRUE) ,                     # this says that the breakpoints that will be used to divide T and lag(q) into different groups are expressed in 
    #                                                                 # raw values of T, and percentiles of the lag(q) distribution
    #                                                                 
    #             breakpts = list(c(0, 2, 4, 6), c(80, 90, 95)) ,     # the first vector says we will divide temperature into five groups: <0 degrees, 2-4 degrees, 4-6 degrees, and above 6 degrees,
    #                                                                 # and we will divide lag(q) into into four groups: the lowest 80 percent, the 80-90th percentiles, the 90-95th percentiles, 
    #                                                                 # and above the 95th percentile.  Thus these breakpoints will construct a total
    #                                                                 # of 20 different sets of precipitation time steps (the four lagged discharge ranges, nested within each of the five
    #                                                                 # temperature ranges).  
    #                                                                 
    #             thresh =  c(-999, 0) ,                              # this says that *if* we calculate breakpoints from percentiles, we evaluate those percentiles among values above the threshold.
    #                                                                 # Thus, for example, the percentiles of lag(Q) will be determined using values of lag(Q) that are above zero.
    #                                                                 # Another example could be if one of the splitting criteria is lagged precipitation, where a substantial fraction of values
    #                                                                 # may be zero (and this fraction will change as the resolution of the time steps changes).
    #                                                                 # Note that criterion values at or below the threshold will still be used in splitting the data set, 
    #                                                                 # just not in setting the values of the breakpoints.
    #                                                                 # The thresh parameter will have no effect on the temperature breakpoints because they are not calculated from percentiles.
    #                                      
    #             by_bin = c(TRUE, TRUE) ,                            # this vector specifies whether the criterion breakpoint percentiles are nested within the bin(s) of the previous criteria.  Thus, for example,
    #                                                                 # the second TRUE means that the 80th percentile breakpoint for lag(Q) is calculated separately for
    #                                                                 # each of the temperature bins (the 80th percentile of precipitation that occurs below 0 degrees C, the 80th percentile that occurs 
    #                                                                 # between 0 and 2 degrees C, and so on).  These may correspond to different absolute
    #                                                                 # values of lag(Q), but they also guarantee that there will be *something* in each set of bins(!).  If the second 
    #                                                                 # element were FALSE instead, then the lag(Q) breakpoints would be the same across all of the temperature bins.
    #                                                                 # The by_bin parameter has no effect if pct_breakpts is FALSE (then the same absolute value of the breakpoint applies across
    #                                                                 # all "parent" bins), and the first element of by_bin never has any effect (because the first criterion isn't nested 
    #                                                                 # within anything) but it must be specified anyhow, so that the by_bin vector is the correct length.
    # ) # end split_crit
    # 
    
    
    
    
    # ERRA optionally divides each precipitation time series into n_sets=(length(breakpts[[1]])+1)*(length(breakpts[[2]])+1)*... interleaved/overlapping subsets
    # by dividing the criteria variables split_params$crit at the stated breakpoint values (if pct_breakpts==FALSE) or at the stated percentile ranks (if pct_breakpts==TRUE).
    # It then creates runoff response distributions for each of these subsets.  Note that the subsets must be analyzed jointly, each with its own RRD coefficients, because
    # the lagged precipitation values from each subset must be interleaved (since filtering by precipitation time produces diagonal stripes in the design matrix for each subset,
    # and thus discharge can depend on the overlapping effects of precipitation from different subsets at different lags).
    # 
    # If xknots is not NULL, nonlinearities in the discharge response to precipitation are analyzed by calculating the runoff response distributions for subsets of
    # each precipitation time series between each successive pair of knot values precipitation (xknots), by calling the nonlinIRF routine.  
    # In this case, the number of columns is *multiplied* by the number of xknots (or number of rows in the xknots matrix) plus one.
    # 
    # This procedure can fail or give nonsense results if the breakpoints and/or xknots are set in incompatible ways.  There must be sufficient cases remaining in each bin delimited by the
    # breakpoints that the regression routine still has enough to work with in each column of the matrix.  The routine reports the minimum number of non-zero
    # values of p in any column (min(n.nz)), which gives a clue about whether the breakpoints and xknots are leaving unreasonably few values to work with.
    
    # Multiple precipitation time series can be analyzed as inputs to the same discharge, although if they are too strongly correlated, it may not be possible
    # to distinguish their effects from one another.  Multiple precipitation time series are supplied by specifying p as a matrix of np columns, rather than 
    # a vector or single-column matrix.  If any element of the splitting parameter "crit" is a matrix, it must be np columns wide, and each column will 
    # be used to split the corresponding column of the precipitation matrix.  Any element of crit that is a vector or a one-column matrix will be applied 
    # to all columns of the precipitation matrix jointly.
    
    # The number of input precipitation series is np=NCOL(p).  The number of subsets these series are divided into is n_sets.  The number of knot points will be nxk=NROW(knots)+1
    # unless xknot_type="even", in which case the number of knot points will be nxk=xknots[1]+1.
    # Each subset of each precipitation series will have m+1 columns representing lag times 0..m.
    # The AR method (for autoregressive noise correction) will also add h more lagtimes, plus h more columns for lagged discharge (where h is the order of the AR correction).  
    # Thus the total number of columns will be np*n_sets*nxk*(1+m+h) + h, plus one column for the constant term.  
    
    # p and q can have missing values (NA).
    # Missing values of q create a missing row in the regression matrix.  Missing values of p create multiple missing rows in the design matrix (diagonals corresponding to each lag).
    # Each criterion (each element of crit) must be same length as vectors of p and q.  Each can have missing values. 
    # Any missing criterion values cause multiple rows in the design matrix to be missing (diagonals corresponding to each lag).
    
    
    
    
    #####################################
    # OUTPUTS FROM ERRA (LINEAR ANALYSES)
    
    # ERRA returns a list as output, with the following objects for linear analyses (i.e., if xknots==NULL)
    #                       where np is the number of input precipitation time series, and
    #                       n_sets is the number of different sets that these are split into, via split_params,
    
    # RRD           data table of runoff response distributions evaluated for each subset of each precipitation variable
    #                        (matrix with np*n_sets and m+1 rows, corresponding to lags 0 through m), and their standard errors
    #                        The RRD table will have one RRD column and one se (standard error) column for each P subset (if P is subsetted for nonstationarity analyses)
    
    # criteria    a data table documenting the splitting criteria, including:
    #           lwr                     lower limit of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    #           upr                     upper limit of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    #           binmean                 average of each criterion in each bin (matrix with np*n_sets rows and n_crit columns)
    
    # peakstats  a data table containing the splitting criteria and the following statistics for each RRD:
    #           set_label                vector of character string labels for each subset
    #           peakht                   peak of RRD by quadratic fitting to uppermost 20% of RRD
    #           peakht_se                standard error
    #           tpeak                    time of peak by quadratic fitting to uppermost 20% of RRD
    #           tpeak_se                 standard error
    #           width                    width of peak, as full width at half maximum
    #           width_se                 standard error
    #           rc                       runoff coefficient (integral of RRD, approximating cumulative discharge per unit precipitation, over the analyzed range of lag times) 
    #           rc_se                    standard error
    
    
    
    
    #####################################
    # OUTPUTS FROM ERRA (NONLINEAR ANALYSES)
    
    # ERRA returns a list as output, with the following objects for nonlinear analyses (i.e., if xknots != NULL)
    #                       where np is the number of input precipitation time series,
    #                       n_sets is the number of different sets that these are split into, via split_params,
    #                       and nxk is the number of different xknots, which will equal the number of specified knot points plus one for the maximum p value.
    #                       Note that the number of xknots and broken-stick segments may be nxk or nxk-1, depending on whether the option show_top_xknot is TRUE or FALSE.
    
    # NRF              data table of Nonlinear Response Functions that express the effect of p on q, averaged over each broken-stick segment between adjacent xknots,
    #                        and their standard errors.  Matrix of np*n_sets*nxk (or nxk-1) columns and m+1 rows, corresponding to lags 0 through m).  
    #                        In contrast to the RRD, the NRF expresses the incremental effect of an additional time step of precipitation at a given rate, rather than
    #                        the effect of an additional unit of volume of precipitation.  So, for example, the NRF for a precipitation intensity of 
    #                        p=20 mm/hr expresses how much an additional rainfall input at 20 mm/hr (over the length of time dt) will raise discharge.  By contrast,
    #                        the RRD would express this same increase in discharge, per unit of precipitation (rather than for the 20mm/hr all together).
    #
    #                        The NRF table will have one NRF column and one se (standard error) column for each P subset and each broken-stick segment.
    #
    #                        For nonlinear analyses, the NRF is used because it shows the shape of the nonlinearity more intuitively than the RRD.
    #                        For example, in a linear system, the NRF will increase linearly with p, whereas RRD would be unchanged across the range of p (within uncertainty).
    #                        And for a quadratically nonlinear system, the NRF will be a parabolic function of p, whereas RRD would increase linearly with p.
    
    # knot_NRF        data table of NRF values evaluated at each nonlinearity knot (i.e., each xknot value).
    #                        Will have one NRF column and one se column for each p subset and each xknot.
    #
    #                 NOTE:  knot_NRF's will typically be noisier than regular NRF's (averaged over each broken-stick segment).  Thus for most purposes,
    #                        NRF's will be preferable to knot_NRF's.
    
    # wtd_avg_RRD     data table of weighted average RRDs, weighted by input p (matrix of np*n_sets columns and m+1 rows) and their standard errors.
    #                        Will have one RRD column and one se column for each p subset.
    
    # peakstats       a data table containing the splitting criteria and the following statistics for each NRF:
    #              
    #           setwm_label              vector of character string labels for each subset
    #           wtd_meanp                weighted mean precipitation within broken-stick segment
    #           pvol                     precipitation volume per time step
    #           tpeak                    time of peak by quadratic fitting to uppermost 20% of NRF
    #           tpeak_se                 standard error
    #           NRF_peakht               peak height of NRF function
    #           NRF_peakht_se            standard error
    #           width                    width of peak, as full width at half maximum
    #           width_se                 standard error
    #           rc                       runoff coefficient (integral of NRF divided by pvol, approximating cumulative discharge over the analyzed range of lag times 
    #                                          that is attributable to each unit of precipitation input at precipitation intensities corresponding to each knot)
    #           rc_se                    standard error
    #           rsum                     runoff volume (integral of NRF, approximating cumulative discharge over the analyzed range of lag times
    #                                          that is attributable to one time unit of precipitation input at precipitation intensity of wtd_meanp).  If P and Q
    #                                          are in mm/h, for example, rsum is also in mm/h, but note that this is mm of *Q* per hour of *P* (at the stated intensity).
    #           rsum_se                  standard error
    #           n.nz                     number of nonzero precipitation time steps in this broken-stick segment
    #
    
    # knot_peakstats  a data table containing the same elements as peakstats, but evaluated at each xknot instead of over each broken-stick segment
    
    # avgRRD_peakstats  a data table containing the peakstats of the weighted average RRD
    
    # criteria  a data table documenting the splitting criteria, including:
    #           setwm_label              vector of character string labels for each subset
    #           wtd_meanp                weighted mean precipitation within broken-stick segment
    #           pvol                     precipitation volume per time step
    
    
    
    
    #####################################
    # OUTPUTS FROM ERRA (BOTH LINEAR AND NONLINEAR ANALYSES)
    
    # options     a string containing values of frequently used ERRA options, to simplify and standardize file naming
    #
    # lagtime     vector of lag times
    #
    # Kbb         covariance matrices for coefficients
    # sets        matrix of length n and width np with integer values (1..n_sets*nxk) indicating which set each time step belongs to, 
    #                       for each precipitation source
    # set_label   set labels including preciptation IDs
    # n           length of original p and q series
    # n.eff       effective sample size, accounting for uneven weighting
    # n.nz        vector of nonzero values (with nonzero weight) in each column of p matrix
    # h           order of AR correction that was applied
    # phi         AR coefficients
    # resid       residuals 
    # resid.acf   autocorrelation function of residuals
    # resid.pacf  partial autocorrelation function of residuals
    # Qcomp       data table comparing observed and fitted values of Q (fitted values will be missing at time steps for which predictors including lagged P are missing)
    #           timestep                 sequence number of time steps
    #           time                     time since start of time series (in time units): timestep*dt
    #           weight                   weight of each time step in fitting procedure.  Where Qfilter==FALSE, weight will be zero and time step will be excluded from 
    #                                    fitting, thus enabling a test of out-of-sample fitting skill
    #           P                        vector or matrix of precipitation inputs
    #           Q                        vector of measured discharge
    #           Qfitted                  vector of fitted discharges
    #           Qresidual                vector of residuals: fitted-measured discharges
  
    
    
    
    
    
    #////////////////////////////////////////////////////////////////////
    # do preliminary checks and conversions, and (optionally) aggregation
    
    TICK <<- Sys.time()                      # start clock (note that TICK is already defined outside this function)
    
    if (is.ts(p)) {                          # if p is a time series, we should convert it to a matrix
      pl <- colnames(p)
      p <- matrix(as.numeric(p), ncol=NCOL(p))  # coerce p to a vector (and then reassemble as a matrix)
      colnames(p) <- pl                         # put column names back
    }
    
    if ((!is.vector(p)) & (!is.matrix(p))) stop("Fatal error in RRD: p must either be, or be conformable to, a matrix or a vector!")
    
    np <- NCOL(p)                           # np is the number of precipitation time series
    
    n <- NROW(p)                            # n is number of precipitation time steps (NROW works for both vectors and matrices)
    
    
    # now we set up the p_label vector
    if (is.null(p_label)||(NROW(p_label)!=np)) {
      if (!is.null(colnames(p))) {         # if the columns of p have names, we use them here (note this overrides p_label)
        p_label <- colnames(p)
      } else {
        if (np==1) {
          p_label <- "p" 
        } else {
          p_label = rep("blank", np)
          for (j in 1:np) p_label[j] <- paste0("p",j)
        }
      }
    }
    
    
    if (is.ts(q)) q <- as.numeric(q)        # if q is a time series, coerce it to a vector
    
    Qfilter <- as.logical(Qfilter)          # coerce Qfilter to a boolean vector just in case it was supplied as something else
    
    if (length(q)!=n |
        length(wt)!=n ) stop("Fatal error in ERRA: input vectors are not all equal in length!")
    
    
    # here we aggregate time steps, if needed
    agg <- ceiling(agg)                     # round aggregation parameter up to next integer
    if (agg<1) agg <- 1                     # values < 1 become 1
    
    if (agg>1) {                                             # don't aggregate unless aggregation parameter is >1
      p <- agg_data(p, agg)                                  # aggregated p
      n <- NROW(p)                                           # aggregated n
      q <- agg_data(q, agg)                                  # aggregated q
      wt <- agg_data(wt, agg)                                # aggregated wt
      split_params$crit <- agg_data(split_params$crit, agg)  # aggregated crit
      Qfilter <- agg_data(Qfilter, agg)                      # aggregated Qfilter
    }
    
    
    # subtract running median (or other quantile) to remove drift or seasonal patterns (if fq>0)
    if (fq>0) {
      if (fq==0.5) qfilt <- stats::runmed(q, k=4*m+1) 
      else if (fq<=1) qfilt <- as.vector(caTools::runquantile(q, k=4*m+1, probs=fq))
      else stop("Fatal error in ERRA: fq must be between 0 and 1, inclusive")
      q <- q - qfilt                                          # subtract running median (or other quantile) to remove drift or seasonal patterns
    }
    
    
    if (nk>2) ser.corr.warn <- 999    # if we are using broken-stick interpolation over lags, suppress serial correlation warning (because it won't work correctly)
    
    
    
    
    
    if ((robust==TRUE) && is.null(xknots)) {  
      warning(paste0("Robust estimation has been selected together with xknots==NULL.  This may result in underestimation of runoff response due to downweighting of high precipitation inputs.  Users are generally advised to use robust estimation only when xknots are also used to account for nonlinear response to precipitation intensity."))
    }
    
    
    
    
    
    
    
    #///////////////////////////
    # make precipitation subsets
    
    ss <- make_sets(n=n, np=np, wt=p, split_params=split_params, p_label=p_label)   # apply criteria to define sets of time steps
    list2env(ss, envir=environment())    # unpack list from make_sets.  
    
    # ss includes:
    #      n_crit    number of subsetting criteria
    #      n_sets    number of different subsets
    #      sets      matrix of length n and width np with integer values (1..n_sets) indicating which set each time step belongs to, for each precipitation source
    #      lwr       lower bounds of each criterion variable (for each set)
    #      upr       upper bounds of each criterion variable (for each set)
    #      binmean   average of each criterion variable (for each set)
    #      set_label vector of np*n_sets strings identifying each set (for each precip source) by its mean values for each criterion
    
    
    p <- as.matrix(p)   # this converts p into a (potentially one-column) matrix, so that what follows will mostly not need to handle np==1 and np>1 as separate cases
    
    p <- ifelse(is.na(sets), NA, p)    # if a p value cannot be assigned to any set, make it NA
    sets[is.na(sets)] <- 1             # change missing set numbers to 1 (this will have no effect, because those rows will have no weight at any lag)
    
    wt[Qfilter==FALSE] <- 0            # here we exclude any rows for which Qfilter==false by setting weight to zero
    
    
    # now split the P variables among the different sets
    p_subsets <- matrix(NA, nrow=n, ncol=(n_sets*np))                                     # create matrix of p filtered for each of the criteria sets
    for (ii in 1:np) 
      for (j in 1:n_sets) 
        p_subsets[ ,((ii-1)*n_sets + j)] <- ifelse((sets[ , ii]==j), p[ , ii], 0)         # we do this here instead of doing it every time we create an entry in the crossproduct matrix
    
    
    
    # now set up xknots and pct_xknots, according to specified xknot_type
    
    if ((!is.null(xknot_type))&&(!is.null(xknots))) {         # skip all this if either xknot_type or xknots is NULL
      if (substr(xknot_type,1,1)=="v") pct_xknots=FALSE       # xknot_type=="values"
      else {if (substr(xknot_type,1,1)=="p") pct_xknots=TRUE  # xknot_type=="percentiles"
      else {if (substr(xknot_type,1,1)=="c") pct_xknots=2     # xknot_type=="cumsum"
      else {if (substr(xknot_type,1,1)=="s") pct_xknots=3     # xknot_type=="sqsum"
      
      else {                                                  # xknot_type=="even"
        if (substr(xknot_type,1,1)!="e") stop("invalid xknot_type specified. valid types are values, percentiles, cumsum, sqsum, or even (v, p, c, s, or e)")
        if (length(xknots)!=2) stop("if xknot_type==even, xknots must contain exactly two values")
        if ((!all(xknots==floor(xknots)))||(xknots[1]<1)||(xknots[2]<1)) stop("if xknot_type==even, xknots must contain two positive integers")
        
        nknots <- xknots[1]                # this is the number of knots we need to set
        mincount <- xknots[2]              # this is the number of points that we need in each interval
        # redefine the xknots vector or matrix so it has a different column for each p_subset, ignoring whatever was originally specified for xknots
        xknots <- matrix(0, nrow=nknots, ncol=NCOL(p_subsets))  
        
        # in setting xknots, we need to exclude P values that will be unusable later (otherwise we could have too few nonzero values in some columns)
        na_flag <- as.integer(is.na(rowSums(p_subsets, na.rm=FALSE)) | is.na(q) | !Qfilter)       # boolean vector of rows where any of p_subsets or q is missing, or q is excluded
        if (sum(na_flag)>0) {                                                          # if there are any of these, we need to propagate them backwards
          if (is.null(h)) maxl <- 1+m+max.AR else maxl <- 1+m+h                         # maximum lag that will need to be canceled
          missing.cancels <- rep(1, maxl)                                              # this is a mask that will be used to propagate the effects of missing values backward
          na_flag <- c(rep(0, (maxl-1)), na_flag)                                      # pad out the beginning of na_flag (with 1's, because the early P values will be unusable too, since their lags don't exist)
          na_flag <- dplyr::lead(na_flag, n=(maxl-1), default=0)                       # shift values toward beginning of vector
          na_flag <- stats::filter(na_flag, missing.cancels, sides=1, circular=TRUE)   # use stats::filter to convolve na_flag forward by maxl, using missing.cancels as a mask
          na_flag <- na_flag[maxl:length(na_flag)]                                     # and remove the padding from the front (this also converts na_flag back to a vector)
          p_flag <- stats::filter(as.integer(is.na(rowSums(p_subsets, na.rm=FALSE))), missing.cancels, sides=1, circular=TRUE)  # also need to convolve missing p values forward
          na_flag <- as.logical(na_flag) | as.logical(p_flag)                          # then we should exclude p values that are flagged by either of these
        } # this could also have been done with stats::convolve, but that can be very slow if any of the prime factors of length(na_flag) are large
          
        # now take each precipitation subset (each column of the precip matrix) in order
        for (j in 1:NCOL(p_subsets)) {
          
          xx <- p_subsets[ ,j]             # copy out each precipitation column
 
          xx <- xx[!na_flag]               # remove values that will eventually correspond to missing rows in the analysis

          xx <- xx[(xx>0.0)]               # remove zero values
          
          xx <- sort(xx, decreasing=TRUE)  # this sort will also remove NA's by default
          
          for (i in 1:nknots) {
            mc <- mincount
            while (xx[mc]==xx[(mc+1)]) mc <- mc+1            # increment mincount until you get a smaller value
            xx_mincount <- 0.5*(xx[mc]+xx[mc+1])             # set knot at mean between xx[mincount] and next smallest value
            xx_threshold <- xx[1]*(nknots-i+1)/(nknots-i+2)  # calculate the fraction of the highest (remaining) p value that would correspond to this knot if knots were equally spaced (from here on down)
            
            xx_threshold <- min(xx_mincount, xx_threshold)   # pick the smaller of these two 
            xknots[(nknots+1-i), j] <- xx_threshold
            
            if (i<nknots) xx <- xx[(xx<=xx_threshold)]  # remove larger values from the sorted list and continue
          }
          
        }
        
        # and now reset pct_xknots, because xknots are now values rather than percentiles
        pct_xknots <- FALSE                                        
      } # end last else
      
     }}}
      
      pct_xknots
      max(xknots)
      if ( (pct_xknots!=FALSE) && (max(xknots)<1) ) xknots <- xknots*100  ## if pct_xknots is not FALSE but max(xknots)<1, assume users have mistakenly specified quantiles and recalculate as percentiles
      
    } # end if
    
    
    
    
    
    
    
    if (verbose) cat(paste0("initialization done...: ", round(difftime(Sys.time(), TICK, units="secs"), 3), "\n"))
    
    
    
    #////////////////////////////////////////////////////
    # now call the impulse response function (IRF) script
    
    if (!is.null(xknots)) dd <- nonlinIRF( y = q, 
                                           x = p_subsets, 
                                           xnames = set_label,
                                           wt = wt,
                                           m = m, 
                                           nk = nk,
                                           xknots = xknots,
                                           pct_xknots = pct_xknots,
                                           nu = nu,
                                           h = h, 
                                           ARprob = ARprob,
                                           ARlim = ARlim,
                                           max.AR = max.AR,
                                           complete = complete.RRD,
                                           verbose = verbose,
                                           robust = robust,
                                           max.chunk = max.chunk,
                                           ser.corr.warn = ser.corr.warn)
    
    else dd <- IRF( y = q, 
                    x = p_subsets, 
                    xnames = set_label,
                    wt = wt,
                    m = m, 
                    nk = nk,
                    nu = nu,
                    h = h, 
                    ARprob = ARprob,
                    ARlim = ARlim,
                    max.AR = max.AR,
                    complete = complete.RRD,
                    verbose = verbose,
                    robust = robust,
                    max.chunk = max.chunk,
                    ser.corr.warn = ser.corr.warn)
    
    if (verbose) cat(paste0("return from IRF...: ", round(difftime(Sys.time(), TICK, units="secs"), 3), " seconds\n"))
    
    
    list2env(dd, envir=environment())    # unpack list from nonlin_IRF  
    
    
    # if we run nonlinIRF, dd includes:
    
    # lags                lags (in number of time steps)
    # xknots              knots for estimation of nonlinear p dependence (matrix of np*n_sets columns by nxk rows)
    # IRF                 nonlinear impulse response function (beta) evaluated at each knot (except zero): matrix of np*n_sets*nxk columns and m+1 rows, corresponding to lags 0 through m)
    # se                  standard errors of IRF coefficients (matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
    # ykx                 nonlinear impulse response, expressed as contribution to q from p, evaluated at each knot (except zero): matrix of np*n_sets*nxk columns and m+1 rows, corresponding to lagtime 0 through m)
    # ykx_se              standard errors of ykx (matrix of np*n_sets*nxk columns and m+1 rows, corresponding to lagtime 0 through m)
    # wtd_avg_IRF         x-weighted average of IRFs 
    # wtd_avg_se          standard error of x-weighted average of IRFs
    # avg_ykx             time-averaged ykx
    # avg_ykx_se          standard error of time-averaged ykx
    # wtd2_avg_IRF        x^2-weighted average of IRF
    # wtd2_avg_se         standard error of x^2-weighted average of IRF
    # wtd_avg_ykx         averaged ykx weighted by input
    # wtd_avg_ykx_se      standard error of weighted average ykx
    # seg_wtd_meanx       weighted mean of x between each pair of knots
    # seg_wtd_IRF         weighted mean of IRF between each pair of knots
    # seg_wtd_se          standard error
    # seg_wtd_ykx         weighted mean ykx between each pair of knots
    # seg_wtd_ykx_se      standard error
    # Kbb                 stack of covariance matrices for each x and each knot (except zero) (array of nx*nxk m+1 by m+1 matrices)
    # n                   length of original x and y series
    # n.eff               effective sample size, accounting for uneven weighting
    # n.nz                number of nonzero values with nonzero weight at each lag of each x variable
    # h                   order of AR correction that was applied
    # phi                 fitted AR coefficients (vector of length h)
    # resid               residuals
    # resid.acf           autocorrelation function of residuals
    # resid.pacf          partial autocorrelation function of residuals
    
    # if we run IRF, dd includes:
    
    # lags                lags (in number of time steps)
    # IRF                 nonlinear impulse response function (beta): matrix of np*n_sets columns and m+1 rows, corresponding to lags 0 through m)
    # se                  standard errors of IRF coefficients (matrix of nx*nxk columns and m+1 rows, corresponding to lags 0 through m)
    # Kbb                 stack of covariance matrices for each x and each knot (except zero) (array of nx*nxk m+1 by m+1 matrices)
    # n                   length of original x and y series
    # n.eff               effective sample size, accounting for uneven weighting
    # n.nz                number of nonzero values with nonzero weight at each lag of each x variable
    # h                   order of AR correction that was applied
    # phi                 fitted AR coefficients (vector of length h)
    # resid               residuals
    # resid.acf           autocorrelation function of residuals
    # resid.pacf          partial autocorrelation function of residuals
    
    
    
    
    
    
    
    
    ################################################################################################
    # here is where we need to mask the uppermost knot, if show_top_xknot is FALSE
    
    
    if ((!is.null(xknots)) && (show_top_xknot==FALSE)) {
      nxk <- NROW(xknots)                        # this will also be re-defined later
      
      for (i in 1:np) for (j in 1:n_sets) {      # build mask for columns/rows to be excluded 
        if ((i==1)&&(j==1)) mask <- nxk else mask <- c(mask, (i-1)*n_sets*nxk + j*nxk)   
      }
      
      # now we mask the elements that pertain to the last xknot
      xknots <- xknots[-nxk, ]
      IRF <- IRF[, -mask]
      se <- se[, -mask]
      ykx <- ykx[, -mask]
      ykx_se <- ykx_se[, -mask]
      n.nz <- n.nz[, -mask]
      seg_wtd_meanx <- seg_wtd_meanx[-nxk ,]
      seg_wtd_IRF <- seg_wtd_IRF[, -mask]
      seg_wtd_se <- seg_wtd_se[, -mask]
      seg_wtd_ykx <- seg_wtd_ykx[, -mask]
      seg_wtd_ykx_se <- seg_wtd_ykx_se[, -mask]
      
    }
    
    
    
    
    #//////////////////////////////////////////////////////////////////
    # now test for low data and warn, fail, or continue, as appropriate
    
    min.nz <- min(n.nz)
    
    # if the minimum number of nonzero precipitation values is below failure threshold, make a hard stop
    if (min.nz<low.data.fail) {   # note that we use warning() rather than stop() here, so that we can return a null result and thus any calling routine cannot continue as normal
      warning(paste0("LOW DATA FAILURE: minimum nonzero precip count at any lag (", min.nz, ") is below user-defined threshold for continuing (", low.data.fail, "). Stopping with null result."))
      return()
    }
    
    # otherwise, if the minimum number of nonzero precipitation values is below warning threshold, issue a warning and continue
    if (min.nz<low.data.warn) warning(paste0("LOW DATA WARNING: minimum nonzero precip count at any lag (", min.nz, ") is below warning threshold (", low.data.warn, ")"))
    
    
    # otherwise continue
    
    
    
    
    list2env(set_lagtime(lags, dt, Qavgd, agg), envir=environment())  # this will create lagtime and first_mult, and update dt, in the current environment  
    
    
    
    #/////////////////////////////////////////////
    # now calculate peakstats and shape the output
    
    
    
    IRF[1,] <- first_mult*IRF[1,]              # RRD for the first lag should adjusted to account for the fraction of q that is sampled *before* the rain in that interval falls
    se[1,] <- first_mult*se[1,]                # for the same reason we need to adjust the uncertainty for the first lag
    
    if (!is.null(xknots)) {                    # if we are taking output from nonlinIRF, we also have to apply first_mult to the other results as well
      wtd_avg_IRF[1,] <- first_mult*(as.matrix(wtd_avg_IRF)[1,])  # double the lag-zero coefficients if q is time-averaged
      wtd_avg_se[1,] <- first_mult*(as.matrix(wtd_avg_se)[1,])
      wtd2_avg_IRF[1,] <- first_mult*(as.matrix(wtd2_avg_IRF)[1,]) 
      wtd2_avg_se[1,] <- first_mult*(as.matrix(wtd2_avg_se)[1,])
      ykx[1,] <- first_mult*(as.matrix(ykx)[1,])  
      ykx_se[1,] <- first_mult*(as.matrix(ykx_se)[1,])
      avg_ykx[1,] <- first_mult*(as.matrix(avg_ykx)[1,])  
      avg_ykx_se[1,] <- first_mult*(as.matrix(avg_ykx_se)[1,])
      wtd_avg_ykx[1,] <- first_mult*(as.matrix(wtd_avg_ykx)[1,])  
      wtd_avg_ykx_se[1,] <- first_mult*(as.matrix(wtd_avg_ykx_se)[1,])
      seg_wtd_IRF[1,] <- first_mult*(as.matrix(seg_wtd_IRF)[1,])
      seg_wtd_se[1,] <- first_mult*(as.matrix(seg_wtd_se)[1,])
      seg_wtd_ykx[1,] <- first_mult*(as.matrix(seg_wtd_ykx)[1,])
      seg_wtd_ykx_se[1,] <- first_mult*(as.matrix(seg_wtd_ykx_se)[1,])
    }
    
    
    
    # note that xknots is now the output from nonlinIRF, which has added a row with the maximum values of each x (unless show_top_xknot=FALSE, in which case this extra row has been removed)
    if (is.null(xknots)||is.null(xknot_type)) nxk <- 1 else nxk <- NROW(xknots)    
    
    RRD = as.matrix(IRF / dt)
    se = as.matrix(se / dt)
    
    
    if (!is.null(xknots)) {
      wtd_avg_RRD <- as.matrix(wtd_avg_IRF / dt)
      wtd_avg_se <- as.matrix(wtd_avg_se / dt)
      wtd2_avg_RRD <- as.matrix(wtd2_avg_IRF / dt)
      wtd2_avg_se <- as.matrix(wtd2_avg_se / dt)
      
      ykx <- as.matrix(ykx / dt)
      ykx_se <- as.matrix(ykx_se / dt)
      avg_ykx <- as.matrix(avg_ykx / dt)
      avg_ykx_se <- as.matrix(avg_ykx_se / dt)
      wtd_avg_ykx <- as.matrix(wtd_avg_ykx / dt)
      wtd_avg_ykx_se <- as.matrix(wtd_avg_ykx_se / dt)
      
      seg_wtd_RRD <- as.matrix(seg_wtd_IRF / dt)
      seg_wtd_se <- as.matrix(seg_wtd_se / dt)
      
      seg_wtd_ykx <- as.matrix(seg_wtd_ykx / dt)
      seg_wtd_ykx_se <- as.matrix(seg_wtd_ykx_se / dt)
    }
    
    # get column labels that combine set_labels and precip breakpoints
    setbp_label <- substring(colnames(IRF), first=5)
    
    if (!is.null(xknots)) colnames(RRD) <- paste0("knot_RRD_", setbp_label) else colnames(RRD) <- paste0("RRD_", setbp_label)
    colnames(se) <- paste0("se_", setbp_label)
    
    # criteria table for each set (but not each nonlinearity breakpoint)
    if (is.null(split_params)) set_crit <- as.data.table(set_label)
    else set_crit <- cbind(as.data.table(set_label), as.data.table(lwr), as.data.table(upr), as.data.table(binmean))
    criteria <- set_crit
    
    if (!is.null(xknots)) {              # if we are analyzing nonlinear response to precipitation, we need to add the information about xknots...
      colnames(avg_ykx) <- paste0("avg_NRF_", set_label)
      colnames(avg_ykx_se) <- paste0("avg_NRF_se", set_label)
      
      colnames(wtd_avg_RRD) <- paste0("wtd_avg_RRD_", set_label)
      colnames(wtd_avg_se) <- paste0("se_", set_label)
      colnames(wtd2_avg_RRD) <- paste0("wtd2_avg_RRD_", set_label)
      colnames(wtd2_avg_se) <- paste0("se_", set_label)
      colnames(wtd_avg_ykx) <- paste0("wtd_avg_NRF_", set_label)
      colnames(wtd_avg_ykx_se) <- paste0("se_", set_label)
      
      colnames(ykx) <- paste0("knot_NRF_", setbp_label)
      colnames(ykx_se) <- paste0("se_", setbp_label)
      
      setwm_label <- substring(colnames(seg_wtd_IRF), first=13)
      colnames(seg_wtd_RRD) <- paste0("RRD_", setwm_label)
      colnames(seg_wtd_se) <- paste0("se_", setwm_label)
      colnames(seg_wtd_ykx) <- paste0("NRF_", setwm_label)
      colnames(seg_wtd_ykx_se) <- paste0("se_", setwm_label)
      
      
      # make criteria table, including nonlinearity breakpoints
      llwr <- matrix(matrix(rep(lwr, nxk), nrow=nxk, byrow=TRUE), ncol=ncol(lwr)) # this replicates each row of lwr, nxk times in sequence
      uupr <- matrix((matrix(rep(upr, nxk), nrow=nxk, byrow=TRUE)), ncol=ncol(upr)) # this replicates each element of upr
      bbinmean <- matrix((matrix(rep(binmean, nxk), nrow=nxk, byrow=TRUE)), ncol=ncol(binmean)) # this replicates each row of binmean
      colnames(llwr) <- colnames(lwr)
      colnames(uupr) <- colnames(upr)
      colnames(bbinmean) <- colnames(binmean)
      if (is.null(split_params)) knot_criteria <- cbind(as.data.table(setbp_label), as.data.table(as.vector(xknots)))
      else knot_criteria <- cbind(as.data.table(setbp_label), as.data.table(llwr), as.data.table(uupr), as.data.table(bbinmean), as.data.table(as.vector(xknots)))
      setnames(knot_criteria, ncol(knot_criteria), "knot")
      knot_criteria[, "knot_pvol" := knot*dt]                # note agg does not appear here, because dt has already been rescaled to account for aggregation
      
      
      # make criteria table, including weighted average precipitation between nonlinearity breakpoints
      if (is.null(split_params)) criteria <- cbind(as.data.table(setwm_label), as.data.table(as.vector(seg_wtd_meanx)))
      else criteria <- cbind(as.data.table(setwm_label), as.data.table(llwr), as.data.table(uupr), as.data.table(bbinmean), as.data.table(as.vector(seg_wtd_meanx)))
      setnames(criteria, ncol(criteria), "wtd_meanp")
      criteria[, "pvol" := wtd_meanp*dt]    # note agg does not appear here, because dt has already been rescaled to account for aggregation
    }
    
    # compile peakstats for RRD
    for (i in 1:np) for (j in 1:n_sets) for (k in 1:nxk) {
      
      this.col <- (i-1)*n_sets*nxk + (j-1)*nxk + k
      
      ps <- peakstats(lagtime=lagtime, bb=RRD[, this.col], se=se[, this.col], Qavgd=Qavgd)   # now get peak height, lag time etc.
      
      if (this.col==1) pss <- as.data.table(ps) else pss <- rbind(pss, as.data.table(ps))  # compile peakstats 
    }
    
    if(!is.null(xknots)) RRDpss <- cbind(knot_criteria, pss) # attach columns of criteria and breakpoints for peakstats
    else RRDpss <- cbind(criteria, pss) # attach columns of criteria and breakpoints for peakstats
    
    if(!is.null(xknots)) {
      
      # replace peakht with NRF, and rename (using the assignment method in data.table)
      RRDpss[, "peakht" := peakht*as.vector(xknots)]
      RRDpss[, "peakht_se" := peakht_se*as.vector(xknots)]
      
      setnames(RRDpss, "peakht", "knot_NRF_peakht")
      setnames(RRDpss, "peakht_se", "knot_NRF_peakht_se")
      
      # add column of runoff volume (runoff coefficient * precipitation)
      RRDpss[, "knot_rsum" := rc*as.vector(xknots)]   
      RRDpss[, "knot_rsum_se" := rc_se*as.vector(xknots)]    
      
      
      
      # compile peakstats for weighted average RRD
      for (i in 1:np) for (j in 1:n_sets) {
        
        this.col <- (i-1)*n_sets + j
        
        ps <- peakstats(lagtime=lagtime, bb=wtd_avg_RRD[, this.col], se=wtd_avg_se[, this.col], Qavgd=Qavgd)   # now get peak height, lag time etc.
        
        if (this.col==1) pss <- as.data.table(ps) else pss <- rbind(pss, as.data.table(ps))  # compile peakstats 
      }
      avgRRDpss <- cbind(set_crit, pss) # attach columns of set criteria
      
      
      
      
      # compile peakstats for seg_wtd_RRD
      for (i in 1:np) for (j in 1:n_sets) for (k in 1:nxk) {
        
        this.col <- (i-1)*n_sets*nxk + (j-1)*nxk + k
        
        ps <- peakstats(lagtime=lagtime, bb=seg_wtd_RRD[, this.col], se=seg_wtd_se[, this.col], Qavgd=Qavgd)   # now get peak height, lag time etc.
        
        if (this.col==1) pss <- as.data.table(ps) else pss <- rbind(pss, as.data.table(ps))  # compile peakstats 
      }
      seg_wtd_RRDpss <- cbind(criteria, pss) # attach columns of criteria and breakpoints for peakstats
      
      # replace peakht with NRF, and rename (using the assignment method in data.table)
      seg_wtd_RRDpss[, "peakht" := peakht*as.vector(seg_wtd_meanx)]
      seg_wtd_RRDpss[, "peakht_se" := peakht_se*as.vector(seg_wtd_meanx)]
      
      setnames(seg_wtd_RRDpss, "peakht", "NRF_peakht")
      setnames(seg_wtd_RRDpss, "peakht_se", "NRF_peakht_se")
      
      # add column of runoff volume (runoff coefficient * precipitation)
      seg_wtd_RRDpss[, "rsum" := rc*as.vector(seg_wtd_meanx)]       
      seg_wtd_RRDpss[, "rsum_se" := rc_se*as.vector(seg_wtd_meanx)] 
      
      
      
      
      
    } #end if !is.null(xknots)
    
    
    
    
    
    
    
    
    
    # compile table comparing measured and fitted Q 
    Qcomp <- data.table(timestep=ycomp$timestep, time=ycomp$timestep*dt, weight=ycomp$wt, P=p[ycomp$timestep[1]:NROW(p),], Q=ycomp$y, Qfitted=ycomp$ypred, Qresidual=ycomp$yresid)
    
    
    if (verbose) cat(paste0("ERRA finished...: ", round(difftime(tock <- Sys.time(), TICK, units="secs"), 3), " seconds\n\n"))
    
    
    
    # compile options string
    op <- ""                                          # start options string
    if (!is.null(split_params)) {                     # if we have splitting parameters, append the labels for the splitting criteria
      op <- paste0(op, "splitby_")
      for (i in 1:length(split_params$crit_label)) op <- paste0(op, split_params$crit_label[i], "_")
    }
    op <- paste0(op, "m=", m)                                 # append the maximum lag m
    if (nk>2) op <- paste0(op, "_nk=", nk)                    # append the number of knots if we are using a broken-stick interpolation
    if (agg>1) op <- paste0(op, "_agg=", agg)                 # append aggregation parameter (if we are aggregating)
    if (h>0) op <- paste0(op, "_h=", h)                       # append h if we do ARMA correction
    if (nu>0) op <- paste0(op, "_nu=", nu)                    # append nu if we are doing regularization
    if (fq>0) op <- paste0(op, "_fq=", fq)                    # append fq if we are using a filter quantile as a baseline
    if (!is.null(xknots)) op <- paste0(op, "_nlin")           # append "nlin" if we are using broken-stick model for nonlinear dependence on precipitation intensity
    if (robust==TRUE) op <- paste0(op, "_robust")             # append "robust" if we are using robust estimation
    
    
    
    
    
    
    if (!is.null(xknots)) {
      return(              # if we are invoking the nonlinear solution method...:
        list(
          
          options = op ,    # a string containing values of many ERRA options, to simplify and standardize file naming
          
          #       # weighted mean RRD within each segment between xknots (data table with one RRD column and one se column for each P subset and interval between xknots)
          #       RRD = cbind(as.data.table(lagtime), as.data.table(seg_wtd_RRD), as.data.table(seg_wtd_se)) ,        !! suppressed in builds after 2023.07.04
          
          #       # runoff response distribution (data table with one RRD column and one se column for each P subset and nonlinearity xknot)
          #       knot_RRD = cbind(as.data.table(lagtime), as.data.table(RRD), as.data.table(se)) ,                    !! suppressed in builds after 2023.07.04
          
          # weighted mean NRF within each segment between xknots (data table with one NRF column and one se column for each P subset and interval between xknots)
          NRF = cbind(as.data.table(lagtime), as.data.table(seg_wtd_ykx), as.data.table(seg_wtd_ykx_se)) ,
          
          # dependence of Q on P (data table with one NRF column and one se column for each P subset and nonlinearity xknot)
          knot_NRF = cbind(as.data.table(lagtime), as.data.table(ykx), as.data.table(ykx_se)) , 
          
          # precipitation-weighted average runoff response distribution (data table with one RRD column and one se column for each P subset)
          wtd_avg_RRD = cbind(as.data.table(lagtime), as.data.table(wtd_avg_RRD), as.data.table(wtd_avg_se)) ,
          
          #       precipitation-weighted average NRF (data table with one NRF column and one se column for each P subset)     !! suppressed in builds after 2024.03.08
          #       wtd_avg_NRF = cbind(as.data.table(lagtime), as.data.table(wtd_avg_ykx), as.data.table(wtd_avg_ykx_se)) ,
          
          #       average NRF (data table with one NRF column and one se column for each P subset)
          #       avg_NRF = cbind(as.data.table(lagtime), as.data.table(avg_ykx), as.data.table(avg_ykx_se)) ,                 !! suppressed in builds after 2023.07.04
          
          peakstats = cbind(seg_wtd_RRDpss, n.nz=colMins(n.nz)) ,     # data table of peak statistics for precipitation-weighted mean response within each broken-stick segment of precipitation intensity
          knot_peakstats = cbind(RRDpss, n.nz=colMins(n.nz)) ,         # data table of peak statistics for RRD and NRF (at each xknot)
          avgRRD_peakstats = avgRRDpss ,    # data table of peak statistics for precipitation-weighted average RRD
          criteria = criteria ,              # data table of upper/lower bounds and bin means for each criterion variable
          Kbb = Kbb ,                         # stack of covariance matrices for each x (array of m+1 by m+1 matrices, one for each Ptime subset)
          sets = sets ,                        # vector of set numbers for each time step
          set_label = set_label ,               # set labels (including precipitation IDs)
          n = n ,                                # original length of input p and q
          n.eff = n.eff ,                         # effective sample size, accounting for uneven weighting
          n.nz = n.nz ,                            # tally of nonzero values with non-zero weight. matrix of (1+n_lagtime) rows and (np*n_sets) columns 
          h = dd$h ,                                # order of AR correction that was applied
          phi = phi ,                                # AR coefficients
          resid = resid ,                             # residuals
          resid.acf = resid.acf ,                      # autocorrelation function of residuals
          resid.pacf = resid.pacf ,                     # partial autocorrelation function of residuals
          Qcomp = Qcomp                                  # data table comparing predicted and observed q values
        )
      ) # end return
      
    } else {
      return(                              # if we are *not* invoking the nonlinear solution method...:
        list(
          
          options = op ,    # a string containing values of many ERRA options, to simplify and standardize file naming
          
          # runoff response distribution (data table with one RRD column and one se column for each P subset)
          RRD = cbind(as.data.table(lagtime), as.data.table(RRD), as.data.table(se)) , 
          
          peakstats = cbind(RRDpss, n.nz=colMins(n.nz)) ,      # data table of peak statistics for RRD
          criteria = criteria ,          # data table of upper/lower bounds and bin means for each criterion variable
          Kbb = Kbb ,                     # stack of covariance matrices for each x (array of m+1 by m+1 matrices, one for each P subset)
          sets = sets ,                    # vector of set numbers for each time step
          set_label = set_label ,           # set labels (including precipitation IDs)
          n = n ,                            # original length of input p and q
          n.eff = n.eff ,                     # effective sample size, accounting for uneven weighting
          n.nz = n.nz ,                        # tally of nonzero values with non-zero weight. matrix of (1+n_lagtime) rows and (np*n_sets) columns 
          h = dd$h ,                            # order of AR correction that was applied
          phi = phi ,                            # AR coefficients
          resid = resid ,                         # residuals
          resid.acf = resid.acf ,                  # autocorrelation function of residuals
          resid.pacf = resid.pacf ,                 # partial autocorrelation function of residuals
          Qcomp = Qcomp                              # data table comparing predicted and observed q values
        ) # end return
      )
    }
    
    
    
  }  #end ERRA

#/////////////////
# END OF ERRA
#/////////////////


