def countTotalBits(sem_string):
    count = 0
    for s in range(len(sem_string)):
        count += 8
    return count


if __name__ == "__main__":
    countTotalBits(sem_string='None')