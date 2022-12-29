library(tidyverse)
library(ranger)

data <- read_csv("data/samples_5m_subset_v1.csv",
    # col_select = matches(".+"),
    col_types = list(
        timestamp = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
        incoming_rate_avg = col_integer(),
        outgoing_rate_avg = col_integer(),
        incoming_rate_max = col_integer(),
        outgoing_rate_max = col_integer(),
        collection_timestamp = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
        src_host = col_character(),
        dst_host = col_character(),
        collection_interval = col_time(format = "%d %* %H:%M:%S")
    ))

data <- data %>%
    as_tibble(index = timestamp) %>%
    filter(timestamp >= "2022-01-01") %>%
    mutate(link = paste(src_host, dst_host, sep = "_")) %>%
    mutate(outgoing_rate_max_log = log(outgoing_rate_max))

data_ts <- data %>%
    select(c("timestamp", "outgoing_rate_max_log", "link"))

# we want links to be in columns so that we can treat them as variables
# so it would be like this
# ts, link1_val1, link2_val1, ... link1_val2, ...

# TODO: investigate duplicate entries in the code below
ts <- data_ts %>%
    pivot_wider(
        id_cols = timestamp,
        names_from = link,
        values_from = outgoing_rate_max_log,
        values_fn = ~head(.x, n = 1) # handle the duplicates
    ) %>%
    drop_na()

ts.new <- select(ts, !timestamp)

coeficients <- list()
# all except the first column - timestamp
for (col in colnames(ts.new)[-1]) {
    print(col)
    frm <- as.formula(paste(col, "~ ."))
    rf <- ranger(formula = frm, data = ts.new, importance = "impurity")
    res <- rf$variable.importance
    coeficients[[length(coeficients) + 1]] <- res
    # add this to a data frame
}


names(coeficients) <- colnames(ts.new)[-1]
my_mat <- do.call(rbind, coeficients)
my_df <- data.frame(id = names(coeficients), my_mat)

f <- my_df %>%
    select(!c(col)) %>%
    pivot_longer(!id) %>%
    mutate(id = as.factor(id)) %>%
    mutate(name = as.factor(name)) %>%
    mutate(value = as.numeric(value))

gg <- f %>%
    ggplot(aes(x = name, y = id)) +
        geom_tile(aes(fill = value, colour = "white"))
show(gg)

# goal: take previous observations in the account
# so we need to include prev timestamps as parameters of our model
# link1_ts1, link2_ts1, ..., link1_ts2, ...
#
# notes:
# - number of embedings = time window = 3h = 36 samples

emb <- embed(as.ts(ts), 36)
colnames(emb) <- rep(colnames(ts), 36)
# TODO: timestamp column is not used at all
emb_data <- data.frame(emb) %>% select(!starts_with("timestamp."))

# TODO: loop over links ... for (col in colnames(ts.new)[-1]) {
rf <- emb_data %>%
    # we dont need self correlation
    select(!starts_with("b6_b1.")) %>%
    select(!starts_with("b1_b6.")) %>%
    ranger(formula = b6_b1 ~ . - timestamp,
    data = ., importance = "impurity")

coef <- rf$variable.importance
print(sort(coef, decreasing = TRUE)[1:10])

importance_df <- data.frame(coef, id = names(coef)) |>
    mutate(id = as_factor(id))

importance_df <- importance_df %>%
    mutate(relative_ts =
        as.numeric(sub(pattern = "^[^\\.]*\\.", id, replacement = ""))) %>%
    mutate(link = sub(pattern = "(\\.[^.]+)$", id, replacement = ""))

my_df$timest[1:15] <- rep(0, 15)

gg <- importance_df %>%
    ggplot(aes(x = relative_ts, y = link, fill = coef)) + geom_tile()
show(gg)