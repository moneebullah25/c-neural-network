CC = gcc
CFLAGS = -Wall -Werror
LDFLAGS = -L ./lib -lc-ann-matrix
SRC_DIR = src
OBJ_DIR = obj
INC_DIR = includes
TEST_DIR = tests
TEST_BIN_DIR = tests/bin
TARGET = libc-ann-matrix

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC))

TESTS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(patsubst $(TEST_DIR)/%.c,$(TEST_BIN_DIR)/%.out,$(TESTS))

.PHONY: all clean

all: $(TARGET) $(TEST_BINS)

$(TARGET): $(OBJ)
	ar rcs lib/$@.a $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I $(INC_DIR) -c $< -o $@

$(TEST_BIN_DIR)/%.out: $(TEST_DIR)/%.c $(TARGET)
	$(CC) $(CFLAGS) -I $(INC_DIR) -o $@ $< $(LDFLAGS) -lm

run_tests: $(TEST_BINS)
	for test in $(TEST_BINS); do \
		./$$test; \
	done

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f lib/*.a
	rm -f $(TEST_BIN_DIR)/*.out
