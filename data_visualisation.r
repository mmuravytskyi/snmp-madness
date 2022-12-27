library(tidyverse)
library(ggplot2)
library(dplyr)
library(shiny)
library(lubridate)

# week starts on Monday
options("lubridate.week.start" = 1)

format_si <- function(...) {
  # Based on code by Ben Tupper
  # https://stat.ethz.ch/pipermail/r-help/2012-January/299804.html

  function(x) {
    limits <- c(1e0, 1e3, 1e6, 1e9)
    prefix <- c("b/s", "kb/s", "Mb/s", "Gb/s")

    # Vector with array indices according to position in intervals
    i <- findInterval(abs(x), limits)

    # Set prefix to " " for very small values < 1e-24
    i <- ifelse(i == 0, which(limits == 1e0), i)

    paste(format(round(x / limits[i], 1),
                 trim = TRUE, scientific = FALSE, ...),
          prefix[i])
  }
}

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

subset <- subset(data, timestamp >= "2022-03-01") %>%
      mutate(diff = incoming_rate_avg - lag(incoming_rate_avg))

d <- subset %>%
  mutate(day = floor_date(timestamp, "day")) %>%
  # TODO: maybe move these to the ggplots
  mutate(week_start = floor_date(timestamp, "week")) %>%
  mutate(week_start_delta = timestamp - week_start) %>%
  mutate(day_start_delta = timestamp - day) %>%
  mutate(link = paste(src_host, "...", dst_host)) %>%
  mutate(week = week(timestamp)) %>%
  mutate(wday = wday(day, week_start = 1, label = TRUE, abbr = FALSE))

gg <- d %>%
  group_by(day, link) %>%
  summarize(avg = mean(incoming_rate_avg),
            src_host = src_host,
            dst_host = dst_host) %>%
  ggplot(aes(x = day, y = avg, colour = link)) +
    geom_line() +
    scale_y_continuous(labels = format_si()) +
    ylab("bitrate") +
    xlab("time") +
    labs(title = "Bitrate statistics averaged by day")

# show(gg)

gg <- d %>%
  group_by(week_start_delta, link) %>%
  summarize(avg = mean(incoming_rate_avg)) %>%
  ggplot(aes(x = week_start_delta, y = avg, colour = link)) +
    geom_line() +
    scale_y_continuous(labels = format_si()) +
    ylab("bitrate") +
    xlab("time") +
    labs(title = "Bitrate statistics averaged by day of the week")

# show(gg)

gg <- d %>%
  group_by(day_start_delta, link) %>%
  summarize(avg = mean(incoming_rate_avg)) %>%
  ggplot(aes(x = day_start_delta, y = avg, colour = link)) +
    geom_line() +
    scale_y_continuous(labels = format_si()) +
    ylab("bitrate") +
    xlab("time [s]") +
    labs(title = "Bitrate statistics averaged by 24h")

plot(gg)