import matplotlib.pyplot as plt
import numpy as np


# insert False to avoid
def show_classes(n_classes,n_train,n_test,X_train,x_test,y_train,y_test,show_image=True):

    if show_image:
        # create figure with subplots
        rows, cols = 4, 12
        fig, ax_array = plt.subplots(rows, cols)
        plt.suptitle('RANDOM SAMPLES FROM TRAINING SET (one for each class)')

        for class_idx, ax in enumerate(ax_array.ravel()):
            if class_idx < n_classes:

                # show a random image of the current class
                cur_X = X_train[y_train == class_idx]
                cur_img = cur_X[np.random.randint(len(cur_X))]
                ax.imshow(cur_img)
                ax.set_title('{:02d}'.format(class_idx))
            else:
                ax.axis('off')

        # hide labels in both axes
        plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
        plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
        plt.draw()

        # bar-chart of classes distribution
        train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)
        for c in range(n_classes):

            # calculate the distribution of training samples over all classes (%)
            train_distribution[c] = np.sum(y_train == c) / n_train

            # calculate the distribution of test samples over all classes (%)
            test_distribution[c] = np.sum(y_test == c) / n_test

        fig, ax = plt.subplots()
        col_width = 0.5
        bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
        bar_test = ax.bar(np.arange(n_classes)+col_width, test_distribution, width=col_width, color='b')

        ax.set_ylabel('PERCENTAGE OF PRESENCE')
        ax.set_xlabel('CLASS LABEL')

        ax.set_title('Classes distribution in traffic-sign dataset')

        ax.set_xticks(np.arange(0, n_classes, 5)+col_width)
        ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])

        ax.legend((bar_train[0], bar_test[0]), ('train set', 'test set'))

        plt.show()

    return

