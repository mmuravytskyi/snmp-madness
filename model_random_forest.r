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
    pivot_wider(id_cols = timestamp,
        names_from = link,
        values_from = outgoing_rate_max_log,
        values_fn = ~head(.x, n = 1) # handle the duplicates
    ) %>%
    drop_na()

coeficients <- list()
# all except the first column - timestamp
for (col in colnames(ts)[-1]) {
    print(col)
    frm <- as.formula(paste(col, "~ ."))
    rf <- ranger(formula = frm, data = ts, importance = "impurity")
    res <- rf$variable.importance
    coeficients[[length(coeficients) + 1]] <- res
    # add this to a data frame
}

names(coeficients) <- colnames(ts)[-1]
my_mat <- do.call(rbind, coeficients)
my_df <- data.frame(id = names(coeficients), my_mat)

f <- my_df %>%
    select(!all_of(col, timestamp)) %>%
    pivot_longer(!id) %>%
    mutate(id = as.factor(id)) %>%
    mutate(name = as.factor(name)) %>%
    mutate(value = as.numeric(value))

gg <- f %>%
    ggplot(aes(x = name, y = id)) +
        geom_tile(aes(fill = value, colour = "white"))
show(gg)
