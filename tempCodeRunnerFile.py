for i in df.select_dtypes(include=[np.number]).columns:
#     sns.boxplot(x=df[i])
#     plt.title(i)
#     plt.show()