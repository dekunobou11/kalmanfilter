# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:17:55 2017
ブラウン運動をする1次元データに対してカルマンフィルタを適用するプログラム
@author: yamazaki
"""
import numpy as np;
import math
import matplotlib.pyplot as plt

def kalmanfilter( A, B, Bu, C, Q, R, u, y, xhat, P ):
    # 線形カルマンフィルタ
    # Input:
    #      A, B, h, C : 対象システムのシステム行列
    #      Q, R : 雑音の共分散行列
    #      u : 状態更新前時点での制御入力 u(k-1)
    #      y : 状態更新後時点での観測出力 y(k)
    #      xhat, P : 更新前の状態推定値 xhat(k-1), 誤差共分散行列P(k-1)
    # Output:
    #      xhat_new : 更新後の状態推定値
    #      P_new    : 更新後の誤差共分散行列
    #      G        : カルマンゲイン
    xhat_new = [];
    P_new    = [];
    G        = [];    

    #numpy形式に変換
    A = np.array( A );        
    B = np.array( B );
    C = np.array( C );
    
    # 事前推定値
    xhatm = A * xhat + Bu * u;
    Pm    = A*P*A.T + B*Q*B.T;
    #カルマンゲイン行列
    G = Pm*C/(C.T*Pm*C+R);
    
    #事後推定値
    xhat_new = xhatm + G*(y-C.T*xhatm);            # 状態
    P_new    = ( np.eye(1) - G*C.T ) * Pm;   # 誤差共分散
    
    #print( xhat_new );
    
    return xhat_new, P_new, G

if __name__ == '__main__':
    ## 問題設定
    A = 1;
    b = 1;
    c = 1;
    Q = 1;
    R = 10;
    N = 300;
    
    ## 観測データの生成
    # 雑音信号の生成
    v = 100*np.random.randn(N,1) + math.sqrt( Q ); # システム雑音
    w = 100*np.random.randn(N,1) + math.sqrt( R ); # 観測雑音
    
    #時系列データの作成
    x = np.zeros( [ N, 1 ] );
    y = np.zeros( [ N, 1 ] );
    
    y[0] = np.array( c ).T * x[0,:] + w[0];
    for k in np.arange(1,N):
        x[k,:] = A * x[k-1,:].T + b*v[k-1];
        y[k]   = np.array( c ).T * x[k,:].T + w[k];
        
    
    ##カルマンフィルタによる状態推定
    xhat = np.zeros( [N, 1 ] );
    
    # 初期値
    P = 0;
    xhat[0,:] = 0;
    
    # 推定値の更新
    for k in np.arange( 1, N ):
        [ xhat[k,:], P, G ] = kalmanfilter( A, b, 0, c, Q, R, 0, y[k], xhat[k-1,:], P);
        
    #print(y)
    print( 'make graph' )
    time = np.arange(0,N).T;
    plt.figure(figsize=(10,6),dpi=80);
    plt.plot(time,y,'k');
    plt.plot(time,x,'r');
    plt.plot(time,xhat,'b');
    plt.show()