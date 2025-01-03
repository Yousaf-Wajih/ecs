CC = g++
C_FLAGS = -MMD -MP -Wall -Wextra -pedantic -std=c++20 -O2 -g
L_FLAGS = -lfmt

BUILD_DIR = build
BIN = $(BUILD_DIR)/ecs-test

SRCS = $(wildcard *cc)
OBJS = $(SRCS:%.cc=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:%.o=%.d)

ARGS =

all: run

run: $(BIN)
	./$(BIN) $(ARGS)

$(BIN): $(OBJS)
	$(CC) $^ -o $@ $(L_FLAGS)

-include $(DEPS)

$(BUILD_DIR)/%.o: %.cc
	mkdir -p $(BUILD_DIR)
	$(CC) $(C_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY : all run clean
