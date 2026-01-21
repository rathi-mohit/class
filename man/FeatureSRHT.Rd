\name{FeatureSRHT}
\alias{FeatureSRHT}
\title{Feature Subsampled Randomized Hadamard Transform}
\usage{
FeatureSRHT(X = NULL, y = NULL, csv = NULL, r = 50, bins = 10, header = FALSE,
            X_test = NULL, y_test = NULL, method = "uniform")
}
\arguments{
  \item{X}{Numeric matrix of predictors.}
  \item{y}{Numeric response vector.}
  \item{csv}{Path to CSV file (alternative to X, y).}
  \item{r}{Integer. Target number of features to select.}
  \item{bins}{Integer. Bins for supervised method discretization.}
  \item{header}{Logical. Whether CSV has a header.}
  \item{X_test}{Optional test matrix for OLS evaluation.}
  \item{y_test}{Optional test response for OLS evaluation.}
  \item{method}{Character vector. Which feature selection strategies to run. 
    Options are \code{"uniform"} (default), \code{"top_r"}, \code{"leverage"}, 
    \code{"supervised"}, or \code{"all"}.}
}
\value{
  A list where the names correspond to the selected methods. Each element contains:
  \itemize{
    \item \code{Train_R2}: R-squared on training data.
    \item \code{Test_R2}: R-squared on test data (if provided).
    \item \code{Indices}: 1-based indices of selected features (in rotated space).
    \item \code{Coefficients}: Full vector of regression coefficients (Intercept at index 1, followed by feature weights). Unselected features have coefficient 0.
  }
}
\description{
  Performs dimensionality reduction using SRHT and selects features based on the 
  specified method(s). Then runs OLS regression on the reduced feature set.
}
\examples{
\dontrun{
  # Run only Uniform (Default)
  res <- FeatureSRHT(X, y, r=50)
  
  # Run Top-r and Leverage
  res <- FeatureSRHT(X, y, r=50, method=c("top_r", "leverage"))
  
  # Run All
  res <- FeatureSRHT(X, y, r=50, method="all")
}
}
