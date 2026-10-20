using Dates

function addSeconds(hhmm::Int, cur_day::Int, cur_month::Int, cur_year::Int, cur_seconds::Int)
    """
    Test function to add seconds to a given time and date.

    # Arguments
    - `hhmm::Int`: Time in 24-hour HHMM format (e.g., 1345 for 1:45 PM).
    - `cur_day::Int`: Current day of the month.
    - `cur_month::Int`: Current month (1-12).
    - `cur_year::Int`: Current year (e.g., 2025).
    - `cur_seconds::Int`: Number of seconds to add.

    # Returns
    - new_year: the year after the seconds were added
    - new_month: the month after the seconds were added
    - new_day: the day after the seconds were added
    - new_hhmm: the hhmm (time in 24-hour HHMM format e.g. 1345 for 1:45 PM) after the seconds were added
    """
    dt = DateTime(cur_year, cur_month, cur_day, fld(hhmm, 100), mod(hhmm, 100), 0)
    dt += Second(cur_seconds)
    new_year = year(dt)
    new_month = month(dt)
    new_day = day(dt)
    new_hhmm = hour(dt) * 100 + minute(dt)
    return new_year, new_month, new_day, new_hhmm
end

function roundTime(hhmm::Int, cur_day::Int, cur_month::Int, cur_year::Int)
    """
    Rounds a time and date to the nearest hour

    # Arguments
    - `hhmm::Int`: Time as an integer in 24-hour HHMM format (e.g., 1345 for 1:45 PM).
    - `cur_day::Int`: Current day of the month.
    - `cur_month::Int`: Current month (1-12).
    - `cur_year::Int`: Current year (e.g., 2025).
    - `cur_seconds::Int`: Number of seconds to add.

    # Returns
    - new_year: the year after the time was rounded
    - new_month: the month after the time was rounded
    - new_day: the day after the time was rounded
    - new_hhmm: the hhmm (time in 24-hour HHMM format e.g. 1345 for 1:45 PM) after the time was rounded
    """
    rounded_time = round(DateTime(cur_year, cur_month, cur_day, fld(hhmm, 100), mod(hhmm, 100), 0), Dates.Hour)
    new_year = year(rounded_time)
    new_month = month(rounded_time)
    new_day = day(rounded_time)
    new_hhmm = hour(rounded_time) * 100 + minute(rounded_time)
    return new_year, new_month, new_day, new_hhmm
end



function monthWord(month::Int)
    """
    Converts a month number (1-12) to its corresponding three-letter abbreviation.

    # Arguments
    - `month::Int`: Month number (1-12).

    # Returns
    - `String`: Three-letter abbreviation of the month (e.g., "Jan", "Feb", etc.).
    """

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    return months[month]
end


function timeString(hhmm::Int)
    """
    Converts a time given in HHMM integer format to a string formatted as "HH:MM".

    # Arguments
    - `hhmm::Int`: Time as an integer in 24-hour HHMM format (e.g., 1345 for 1:45 PM).

    # Returns
    - `String`: The time formatted as "HH:MM" (e.g., "13:45").
    """
    hour = fld(hhmm, 100)   # Extract the hour
    minute = mod(hhmm, 100) # Extract the minute
    return string(lpad(hour, 2, '0'), ":", lpad(minute, 2, '0'))  # Format as "HH:MM"
end
