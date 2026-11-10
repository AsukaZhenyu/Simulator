// SHA-256 without a platform crypto provider.
//
// `components.hpp` and `initial_assets.hpp` digest copied and transported bytes
// so a reader can compare the two sides without the values. On Windows that
// digest came from BCrypt, which was the one Win32 dependency with no POSIX
// counterpart in the C library. A digest is a pure function of its input: the
// same bytes produce the same 64 hex characters on every host, so one
// implementation here is what makes a Linux receipt and a Windows receipt
// comparable at all -- and it removes the `bcrypt` link rather than adding a
// second provider on the other side.
//
// `Hasher` is incremental because one caller digests a multi-gigabyte initial
// asset file and must not hold a second copy of it. `digest_hex` is the one-shot
// form for buffers already in memory.
//
// The known-answer vectors in `probe.cpp`'s self-test are what guard this: the
// empty string, `abc`, and a 56-byte message whose 0x80 and length bytes do not
// fit in one final block, so neither the length encoding nor the block loop can
// be wrong and still pass.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace portable_sha256 {

inline std::uint32_t rotate_right(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

inline void compress(std::uint32_t state[8], const std::uint8_t *block) {
    static const std::uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::uint32_t schedule[64];
    for (int index = 0; index < 16; ++index)
        schedule[index] = (std::uint32_t(block[index * 4]) << 24)
                          | (std::uint32_t(block[index * 4 + 1]) << 16)
                          | (std::uint32_t(block[index * 4 + 2]) << 8)
                          | std::uint32_t(block[index * 4 + 3]);
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t first = rotate_right(schedule[index - 15], 7)
                                    ^ rotate_right(schedule[index - 15], 18)
                                    ^ (schedule[index - 15] >> 3);
        const std::uint32_t second = rotate_right(schedule[index - 2], 17)
                                     ^ rotate_right(schedule[index - 2], 19)
                                     ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16] + first + schedule[index - 7] + second;
    }
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t choice = (e & f) ^ (~e & g);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t first = h + sum1 + choice + constants[index] + schedule[index];
        const std::uint32_t second = sum0 + majority;
        h = g; g = f; f = e; e = d + first; d = c; c = b; b = a; a = first + second;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

class Hasher {
public:
    void update(const void *data, std::size_t size) {
        const std::uint8_t *bytes = static_cast<const std::uint8_t *>(data);
        total_ += size;
        if (buffered_ != 0) {
            const std::size_t take = size < 64 - buffered_ ? size : 64 - buffered_;
            std::memcpy(buffer_ + buffered_, bytes, take);
            buffered_ += take;
            bytes += take;
            size -= take;
            if (buffered_ < 64) return;
            compress(state_, buffer_);
            buffered_ = 0;
        }
        while (size >= 64) {
            compress(state_, bytes);
            bytes += 64;
            size -= 64;
        }
        if (size != 0) {
            std::memcpy(buffer_, bytes, size);
            buffered_ = size;
        }
    }

    // Idempotent: it finalizes copies, so a caller can digest a snapshot and
    // keep feeding the running hash if it ever needs to.
    std::string hex() const {
        std::uint32_t state[8];
        std::memcpy(state, state_, sizeof(state));
        std::uint8_t buffer[64];
        std::memcpy(buffer, buffer_, sizeof(buffer));
        std::size_t buffered = buffered_;
        buffer[buffered++] = 0x80;
        if (buffered > 56) {
            std::memset(buffer + buffered, 0, 64 - buffered);
            compress(state, buffer);
            buffered = 0;
        }
        std::memset(buffer + buffered, 0, 64 - buffered);
        const std::uint64_t bits = total_ * 8ULL;
        for (int index = 0; index < 8; ++index)
            buffer[63 - index] = static_cast<std::uint8_t>(bits >> (8 * index));
        compress(state, buffer);
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::uint32_t value : state) out << std::setw(8) << value;
        return out.str();
    }

private:
    std::uint32_t state_[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::uint8_t buffer_[64] = {};
    std::size_t buffered_ = 0;
    std::uint64_t total_ = 0;
};

// The digest of `size` bytes at `data`, as lowercase hex. A null `data` with a
// non-zero size is the caller's error; the zero-length message is legal.
inline std::string digest_hex(const void *data, std::size_t size) {
    Hasher hasher;
    hasher.update(data, size);
    return hasher.hex();
}

}  // namespace portable_sha256
