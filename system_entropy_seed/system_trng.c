/**
 * DISCLAIMER:
 * These functions should not be used in any security-sensitive context or wherever high-quality randomness is required.
 * For cryptographic purposes or any application where high-quality random numbers are needed, it is
 *     _ strongly recommended _
 * to use a cryptographically secure random number generator (CSPRNG) such as
 *     - /dev/urandom on Unix-like systems
 *     - or CryptGenRandom on Windows.
 *
 * The functions in this file are only intended to demonstrate how to gather entropy from various sources on a system without any user interaction.
 */

/**
 * INSTRUCTIONS:
 * gcc system_trng.c -o system_trng -lm
 * ./system_trng
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>

// The following headers are only available on Linux
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>

// # Which sources of entropy are available on a system without user interaction?

/**
 * @brief Retrieves the current system time in nanoseconds.
 *
 * The function utilizes the clock_gettime system call to obtain the current time with high resolution
 * in nanoseconds. The total number of nanoseconds since the epoch is computed by multiplying the
 * seconds part by 1,000,000,000 and adding the nanoseconds part.
 *
 * @note This function should not be used in any security-sensitive context where high-quality
 * randomness is required, as the output is based on the current time and is predictable.
 *
 * @return uint64_t representing the current time in nanoseconds since the epoch.
 */
uint64_t get_time_in_nanoseconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/**
 * @brief Retrieves the current CPU Time Stamp Counter (TSC) in cycles.
 *
 * The function utilizes inline assembly with the `cpuid` and `rdtsc` instructions to serialize
 * instruction execution and read the Time Stamp Counter (TSC), respectively. The `cpuid`
 * instruction is used for serialization to ensure that all preceding instructions are completed
 * before reading the TSC. The `rdtsc` instruction reads the TSC into registers EDX:EAX. The lower
 * 32 bits are in EAX, and the upper 32 bits are in EDX.
 *
 * The output value is the full 64-bit TSC obtained by combining the upper and lower 32 bits.
 *
 * @note This function is highly platform-dependent and should be used with caution.
 * @note The TSC can vary depending on the CPU frequency, which might change due to power saving
 * modes or turbo boost.
 * @note The function uses inline assembly and is specific to x86 architecture.
 * @note This function should not be used in a security-sensitive context where high-quality
 * randomness is required, as the output is based on the current CPU cycles and can be predictable.
 *
 * @return uint64_t representing the current value of the TSC in cycles.
 */
uint64_t get_cpu_cycles()
{
    unsigned int lo, hi;

    // Inline assembly to serialize instruction execution and read the TSC
    __asm__ __volatile__(
        "cpuid\n\t"          // Serialize instruction execution
        "rdtsc\n\t"          // Read TSC
        : "=a"(lo), "=d"(hi) // Output operands
        :
        : "%ebx", "%ecx"); // Clobbered registers

    // Return the full 64-bit TSC
    return ((uint64_t)hi << 32) | lo;
}

/**
 * @brief Retrieve the time taken to access the disk in microseconds.
 *
 * The function measures the time taken to perform a read operation from the file "/dev/urandom".
 * It uses the clock_gettime system call for high-resolution timing. The total number of
 * microseconds taken to read a character from the file is calculated as the difference between
 * the timestamps before and after the read operation.
 *
 * This function primarily serves as a performance measurement tool for profiling disk access times
 * and should not be used in production settings for critical applications.
 *
 * @note This function opens a file but does not utilize its contents.
 * @note The function assumes that "/dev/urandom" exists and can be read.
 * @note The accuracy of the measurement is subject to various factors including system load,
 * disk type, and file system.
 * @note This function may fail if the file cannot be opened, but it does not handle such errors explicitly.
 *
 * @return uint64_t representing the time taken in microseconds to perform a read operation from the disk.
 */
uint64_t get_disk_access_time()
{
    // Open the file. Note that the contents are not used.
    FILE *file = fopen("/dev/urandom", "r");
    if (!file)
    {
        return 0; // Return 0 in case the file could not be opened.
    }

    char ch;
    struct timespec start_time, end_time;

    // Record the start time
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Perform the read operation
    fread(&ch, sizeof(ch), 1, file);

    // Record the end time
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Close the file
    fclose(file);

    // Calculate and return the time taken in microseconds
    return (end_time.tv_sec - start_time.tv_sec) * 1000000ULL +
           (end_time.tv_nsec - start_time.tv_nsec) / 1000ULL;
}

/**
 * @brief Retrieves the system's one-minute load average as a fixed-point 64-bit value.
 *
 * The function retrieves the one-minute load average of the system, which is a measure of
 * the amount of computational work that the system has been performing. The load average
 * is conventionally represented as a floating-point number with 32 fractional bits,
 * where a value of 1.0 is roughly equivalent to full utilization of one CPU core.
 *
 * @note This representation is more suitable for environments where floating-point arithmetic
 * is not desirable. To convert the fixed-point representation back to a floating-point number,
 * divide the result by 2^32.
 * @example Optionally, convert back to floating-point for display
 * ```c
    double load_avg = (double)fixed_point_load_avg / (1ULL << 32);
    printf("One-minute load average: %f\n", load_avg);
   ```
 *
 * @note This function uses the getloadavg() library call, which may not be available on all systems.
 * @note Load average can be influenced by various factors including system load, number of CPU cores,
 * and the specific behavior of the operating system's scheduler.
 *
 * @return uint64_t representing the one-minute load average as a fixed-point number with 32 fractional bits.
 *         Return 0 if the retrieval fails.
 *
 */
uint64_t get_system_load_avg()
{
    double loadavg[1];

    // Retrieve the one-minute load average
    if (getloadavg(loadavg, 1) == -1)
    {
        printf("One-minute load average failed: %f\n", -1.0);
        return 0; // Return 0 if retrieval fails
    }
    if (loadavg[0] == 0)
    {
        printf("One-minute load average failed: %f\n", loadavg[0]);
        return 0; // Return 0 if the load average is 0
    }

    // Convert the load average to a fixed-point 64-bit value
    return (uint64_t)(loadavg[0] * (1ULL << 32));
}

/**
 * @brief Retrieves the network ping time to the address 8.8.8.8 in microseconds as a 64-bit value.
 *
 * The function uses the `ping` command to send a single ICMP echo request to the IP address 8.8.8.8
 * (a public DNS server operated by Google) and measures the round-trip time. The `ping` command's
 * output is processed to extract the round-trip time, which is returned in microseconds as a
 * 64-bit unsigned integer.
 *
 * This function can be used for basic network latency measurements. However, it is important to
 * note that this approach involves executing a shell command, which can have security implications
 * and may not be suitable for all environments or use cases.
 *
 * @note This function depends on the availability of the `ping` command and assumes it is installed
 * and accessible in the system's PATH.
 * @note The function assumes that the output format of `ping` matches the expected pattern. Changes
 * in the output format may cause incorrect results.
 * @note ICMP packets used by `ping` can be filtered by firewalls, affecting the results.
 * @note The execution time of this function can vary depending on network conditions.
 * @note The function uses shell command execution and should be used with caution, especially in
 * security-sensitive contexts.
 *
 * @return uint64_t representing the network ping time in microseconds to the address 8.8.8.8.
 *         Return 0 if the retrieval fails or in case of an error.
 */
uint64_t get_network_ping_time()
{
    // Execute the ping command and capture the output
    FILE *ping_response = popen("ping -c 1 8.8.8.8 | awk -F '/' 'END {print $(NF-2)}'", "r");
    if (!ping_response)
    {
        return 0; // Return 0 in case of an error
    }

    char buffer[32];
    fgets(buffer, sizeof(buffer), ping_response);
    pclose(ping_response);

    // Convert the extracted round-trip time to microseconds and return as a 64-bit value
    double ping_time_seconds = atof(buffer);
    return (uint64_t)(ping_time_seconds * 1000000.0);
}

/**
 * @brief Retrieves the system's uptime in seconds as a 64-bit value.
 *
 * The function utilizes the sysinfo() system call to obtain various system statistics,
 * including the uptime, which is the time duration for which the system has been running
 * since it was last booted.
 *
 * The uptime is reported in seconds and is represented as a 64-bit unsigned integer,
 * allowing it to accurately report the uptime for systems that have been running for
 * extremely long durations.
 *
 * @note This function uses the sysinfo() system call, which may not be available on all systems.
 * @note The sysinfo() system call provides the uptime as a long integer; this function casts
 *       the uptime to a 64-bit unsigned integer.
 *
 * @return uint64_t representing the system's uptime in seconds. Returns 0 if sysinfo() fails.
 */
uint64_t get_uptime()
{
    struct sysinfo sys_info;

    // Retrieve system information
    if (sysinfo(&sys_info) != 0)
    {
        return 0; // Return 0 if sysinfo() fails
    }

    // Return the uptime in seconds as a 64-bit value
    return (uint64_t)sys_info.uptime;
}

/**
 * @brief Retrieves the CPU time used by the process during a PID-dependent CPU-bound operation in microseconds.
 *
 * This function obtains the process ID (PID) using getpid() and then executes a CPU-bound operation in a loop.
 * The number of iterations of the loop is determined by the PID and the current value of the clock.
 * The function measures the CPU time consumed by this operation in microseconds using the clock() function.
 *
 * This function can be used for analyzing the relationship between the CPU time consumed and the PID.
 * However, it is important to note that the actual CPU time consumed can be influenced by various factors
 * including system load, scheduling, and available resources.
 *
 * @note This function uses the clock() function, which measures CPU time used by the process. This is
 * different from wall-clock time.
 * @note The number of loop iterations is dependent on the PID and current value of the clock, and hence
 * can vary for different executions.
 * @note This function performs an artificial CPU-bound operation for measurement purposes and doesn't
 * have a practical application.
 *
 * @return uint64_t representing the CPU time used by the PID-dependent operation in microseconds.
 */
uint64_t pid_clock()
{
    pid_t pid = getpid();
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    for (int i = 0; i < (pid * 1000 + start % 100); i++)
    {
        // Perform a CPU-bound operation that depends on the PID and the loop index
        double d = (time(NULL) % 2 == 0) ? i * i : i * i * i;
    }
    end = clock();

    // Convert CPU time used to microseconds
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;

    // Return the CPU time used as a 64-bit value
    return (uint64_t)cpu_time_used;
}

// # How can we evaluate the entropy of the given sources?

// Toy CSPRNG using Xorshift algorithm
uint64_t xorshift(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

// Perform Chi-squared Test
void perform_chi_squared_test(const char *source_name, uint64_t seed, size_t n)
{
    int buckets[256] = {0};
    double expected = n / 256.0;
    double chi_squared = 0.0;
    uint64_t random_data[n];

    // Generate random data using the toy CSPRNG
    for (size_t i = 0; i < n; i++)
    {
        random_data[i] = xorshift(&seed);
    }

    // Count the occurrences in each bucket
    for (size_t i = 0; i < n; i++)
    {
        buckets[random_data[i] & 0xFF]++;
    }

    // Calculate the Chi-squared statistic
    for (int i = 0; i < 256; i++)
    {
        chi_squared += pow(buckets[i] - expected, 2) / expected;
    }

    // Degrees of freedom is 256 - 1 = 255
    // For a significance level of 0.05, the critical value is 293.25
    int passed = chi_squared < 293.25;
    printf("%s %s: %f. The output is %sconsistent with a random distribution.\n", passed ? "[v]" : "[x]", source_name, chi_squared, passed ? "" : "not ");
}

// # THE RANKINGS
// Network Ping Time   (232.384) - This source has the lowest Chi-squared statistic among those that are consistent with a random distribution. This implies that it may be a good source of entropy, although it is highly environment-dependent and can be affected by network conditions.
// Time in Nanoseconds (246.208) - This source shows a relatively low Chi-squared statistic and is consistent with a random distribution. It represents the lower bits of the current time in nanoseconds, which is expected to be fairly random and unpredictable.
// Disk Access Time    (247.744) - Similar to the Time in Nanoseconds, this source is also consistent with a random distribution. It measures the time it takes to access the disk. The entropy in this source may come from the variability in disk access times, though it may also be environment-dependent.
// CPU Cycles          (261.568) - This source is consistent with a random distribution but has a slightly higher Chi-squared statistic. It is based on the CPU's Time Stamp Counter, which should be fairly unpredictable.
// System Uptime       (273.344) - Though consistent with a random distribution, it has a higher Chi-squared statistic. It represents the time the system has been up and running, so it is more predictable and changes slowly.
// PID Clock           (293.312) - This source has the highest Chi-squared statistic among the sources and is sometimes still marked as consistent with random distribution, indicating that it's less random compared to the others. It's based on the CPU clock, but it involves a CPU-bound loop that might introduce patterns.
// System Load Average (-------) - Failed horribly and repeatedly. It represents the system load average, which is expected to be fairly predictable and changes slowly.

int main()
{
    // Define the array of seeds gathered from different sources
    struct
    {
        const char *source_name;
        uint64_t (*source_function)();
    } entropy_sources[] = {
        {"Time in Nanoseconds", get_time_in_nanoseconds},
        {"CPU Cycles", get_cpu_cycles},
        {"Disk Access Time", get_disk_access_time},
        {"System Load Average", get_system_load_avg},
        {"Network Ping Time", get_network_ping_time},
        {"System Uptime", get_uptime},
        {"PID Clock", pid_clock}};

    // Define the number of random numbers to generate for each seed
    size_t n = 1000;

    // Perform the Chi-squared test for each seed individually
    printf("Chi-squared test for each seed individually:\n");
    for (size_t i = 0; i < sizeof(entropy_sources) / sizeof(entropy_sources[0]); i++)
    {
        uint64_t seed = entropy_sources[i].source_function();
        perform_chi_squared_test(entropy_sources[i].source_name, seed, n);
    }

    return 0;
}