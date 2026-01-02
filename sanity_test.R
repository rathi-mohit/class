library(Rcpp)
library(testthat)

sourceCpp("src/IBOSS_wrapper.cpp")
source("R/IBOSS.R")

test_that("Random Sanity Check Test", {
  X = matrix(rnorm(20), 500000, 100)
  y = rnorm(500000)

  start <- Sys.time()
  res <- IBOSS(X, y, k = 50000)
  end <- Sys.time()

  expect_true(nrow(res$X_selected) <= 50000)
  cat(nrow(res$X_selected))
  cat("\nTime taken:", end - start, "\n")
})

test_that("CSV Input Test", {
  X = matrix(rnorm(20), 50, 4)
  y = rnorm(50)
  dat = data.frame(X, y)
  write.csv(dat, "temp_test_data.csv", row.names = FALSE)

  res1 <- IBOSS(X = X, y = y, k = 20)
  res2 <- IBOSS(csv = "temp_test_data.csv", k = 20, header = TRUE)

  expect_equal(res1$X_selected, res2$X_selected)
  expect_equal(res1$y_selected, res2$y_selected)

  file.remove("temp_test_data.csv")
})
