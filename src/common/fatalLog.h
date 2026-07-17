#ifndef KYTY_COMMON_FATAL_LOG_H_
#define KYTY_COMMON_FATAL_LOG_H_

namespace Common {

// Optional mirror path (e.g. --printf-output-file). Always appends to _kyty_fatal.txt as well.
void SetFatalLogMirrorPath(const char* path);

void LogFatalToFile(const char* message);

} // namespace Common

#endif /* KYTY_COMMON_FATAL_LOG_H_ */
