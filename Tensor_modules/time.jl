function bisesto(year)
	if year%4 == 0
        	if year%100 == 0
                	if year%400 == 0
                        	days = 366
                	else
                        	days = 365
                	end
        	else
                	days = 366
        	end
	else
        	days = 365
	end
	return days
end

function findyear(number_of_days,year)

	days = bisesto(year)
	number_of_days = number_of_days - days

	if number_of_days <= 0
		return year, (number_of_days+days)
	else
		year = year + 1
		findyear(number_of_days,year)
	end
end

function findmonths(actual_year,x,remaining_days)

	if (x==4||x==6||x==9||x==11)
		flag = remaining_days - 30
	elseif (x==2)
		if bisesto(actual_year)==365
			flag = remaining_days - 28
		else
			flag = remaining_days - 29
		end
	else
		flag = remaining_days - 31
	end

	if (flag)<=0
		month = x
		return remaining_days, month
	else
		x = x+1
		remaining_days = flag
		findmonths(actual_year,x,remaining_days)
	end
end

function print_time()
actual_time = time()

year = 1970
seconds_per_minute = 60
seconds_per_hour   = seconds_per_minute*60
seconds_per_day    = seconds_per_hour*24

number_of_days    = actual_time÷seconds_per_day
remaining_seconds = actual_time%seconds_per_day

x = 1

actual_year, remaining_days  = findyear(number_of_days,year)
actual_day, actual_month = findmonths(actual_year,x,remaining_days+1)

actual_hour             = remaining_seconds÷seconds_per_hour
actual_minute           = (remaining_seconds%seconds_per_hour)÷seconds_per_minute
actual_second           = (remaining_seconds%seconds_per_hour)%seconds_per_minute

print(trunc(Int64,actual_day),"/",actual_month,"/",actual_year," ")
print(trunc(Int64,actual_hour),":",trunc(Int64,actual_minute),":",trunc(Int64,actual_second)," (GMT)")
end
