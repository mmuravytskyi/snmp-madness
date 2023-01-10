library(tidyverse)
library(ranger)
library(lubridate)

data <- read_csv("data/samples_5m_subset_v1.csv",
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
    mutate(incoming_rate_avg_log = log(incoming_rate_avg))

data_ts <- data %>%
    select(c("timestamp", "incoming_rate_avg_log", "link")) %>%
    pivot_wider(
        id_cols = timestamp,
        names_from = link,
        values_from = incoming_rate_avg_log,
        values_fn = ~head(.x, n = 1) # handle the duplicates
    ) %>%
    drop_na()

# we want links to be in columns so that we can treat them as variables
# so it would be like this
# ts, link1_val1, link2_val1, ... link1_val2, ...
USE_TIME_DIMENTION <- TRUE

if (!USE_TIME_DIMENTION) {
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
}

# goal: take previous observations in the account
# so we need to include prev timestamps as parameters of our model
# link1_ts1, link2_ts1, ..., link1_ts2, ...
#
# notes:
# - number of embedings = time window = 3h = 36 samples

emb <- embed(as.ts(data_ts), 24)
colnames(emb) <- rep(colnames(data_ts), 24)
emb_data <- data.frame(emb) %>% select(!starts_with("timestamp."))

# LIMITATIONS
# - we only use 100 trees, maybe it's enough maybe it's not, I don't know
# - we only use data starting form 2022.01.01

# -1 is the index of `timestamp` column
for (colname in colnames(data_ts)[-1]) {
    sprintf("Building RF for %s ...", colname)
    src_str <- sub(pattern = "(\\_[^_]+)$", colname, replacement = "")
    dst_str <- sub(pattern = "^[^_]*\\_", colname, replacement = "")

    frwrd_link_str <- paste(src_str, dst_str, sep = "_")
    bkwrd_link_str <- paste(dst_str, src_str, sep = "_")

    formula_str <- paste(frwrd_link_str, "~ . - timestamp")
    sprintf("Formula: %s", formula_str)
    rf <- emb_data %>%
        # we dont need self correlation
        select(!starts_with(bkwrd_link_str)) %>%
        select(!starts_with(paste(frwrd_link_str, ".", sep = ""))) %>%
        ranger(formula = as.formula(formula_str),
        data = ., importance = "impurity", num.trees = 100)

    coef <- rf$variable.importance
    print(rf$r.squared)

    importance_df <- data.frame(coef, id = names(coef)) %>%
        mutate(id = as_factor(id),
            relative_ts =
            sub(pattern = "^[^\\.]*\\.", id, replacement = ""),
            link = sub(pattern = "(\\.[^.]+)$", id, replacement = ""),
            relative_ts = as.numeric(relative_ts) * 5) %>%
            replace_na(list(relative_ts = 0))

    # TODO: make this plot a bit more pleasing
    gg <- importance_df %>%
        ggplot(aes(x = relative_ts, y = link, fill = coef)) +
        geom_tile() +
        theme_bw() +
        scale_fill_viridis_c(option = "B", direction = -1) +
        labs(
            x = "Time window [min]",
            y = "Links [source_destination]",
            fill = "Importance",
            title = paste("Spatial and temporal importance of variables for", 
                frwrd_link_str)
        )

    plot_fl_path <- paste("plots", frwrd_link_str, sep = "/")
    plot_fl_path <- paste(plot_fl_path, "png", sep = ".")
    ggsave(plot_fl_path)
}