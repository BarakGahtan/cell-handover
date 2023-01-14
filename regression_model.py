import time
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.express as px


def dec_tree(df_with_lost_frames, shift, m):
    X = df_with_lost_frames.drop(columns={'switchover'}, axis=1).values
    y = df_with_lost_frames['switchover'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a decision tree regressor model on the training data
    t0 = time.time()
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    t1 = time.time()
    # Use the model to make predictions on the test data
    predictions = model.predict(X_test)
    t2 = time.time()
    mae = mean_absolute_error(predictions, y_test)
    print('took', round(t1 - t0, 4), 'to create the tree', ' took ', round(t2 - t1, 4), ' to infer', '   mae:', mae)
    print('feature importance')

    feature_names = list(df_with_lost_frames.drop(columns={'switchover'}, axis=1).columns)
    feaure_imp = model.feature_importances_
    feat_imp_df = pd.DataFrame(index=feature_names, data=feaure_imp)
    feat_imp_df.to_csv('feat_imp_modem_' + str(m) + '_plus_' + str(shift) + '_' + str(shift) + '.csv')
    plot1_df = pd.DataFrame(columns={'real', 'pred'})
    plot1_df['real'] = y_test
    plot1_df['pred'] = predictions

    px.line(plot1_df).write_html('real_vs_pred_dec_tree' + str(m) + '_plus_' + str(shift) + '.html')
    return model
