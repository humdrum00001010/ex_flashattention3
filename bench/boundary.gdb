set pagination off
set confirm off
set breakpoint pending on
set print thread-events off
handle SIGPIPE nostop noprint pass
handle SIGUSR1 nostop noprint pass
handle SIGUSR2 nostop noprint pass
set $run_count = 0
set $fa3_count = 0

break exla::run_io
commands
  silent
  set $run_count = $run_count + 1
  continue
end

break exla_fa3_forward
commands
  silent
  set $fa3_count = $fa3_count + 1
  continue
end

run
printf "GDB_RUN_IO_COUNT=%d GDB_FA3_COUNT=%d\n", $run_count, $fa3_count
quit
