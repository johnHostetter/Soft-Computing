library(RoughSets)
data(RoughSetData)
hiring.data <- RoughSetData$hiring.dt

setwd("/home/john/PycharmProjects/soft_computing/rough")
flc_rules <- read.csv("rules.csv")
flc_rules <- subset(flc_rules, select=-c(X))
# flc_rules <- head(flc_rules, 1000)

decision.table <- SF.asDecisionTable(dataset = flc_rules, decision.attr = 130, indx.nominal = rep(c(TRUE), 130))
#disc.matrix <- BC.discernibility.mat.RST(decision.table, return.matrix = TRUE)
## discretization:
#cut.values <- D.discretization.RST(decision.table,
#                                   type.method = "local.discernibility",
#                                   maxNOfCuts = 1)
#decision.table <- SF.applyDecTable(decision.table, cut.values)
rules <- RI.LEM2Rules.RST(decision.table)
