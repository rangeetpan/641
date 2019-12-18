x1=[2.67
3.33
2.33
3.77
2.05
3.37
3.14
3.31
2.27
1.00
1.68
1.98
2.35
1.98
3.59
3.22
2.65
2.73
3.06
2.41
2.09
1.56
1.92
2.00
1.22
1.45
2.42
3.80
3.05
2.43
3.34
3.08
2.67
3.30
1.71
1.71
1.82
2.10
3.09
3.20
2.08
3.66
1.43
1.90
2.28
2.31
2.34
1.64
2.83
2.12
3.66];
x2=[2.61
-0.43
0.68
2.26
1.69
0.92
-0.38
0.83
0.30
0.91
1.32
0.79
0.79
0.45
1.89
1.35
0.51
0.50
-1.37
0.85
1.08
1.58
2.40
2.01
1.71
-0.49
1.37
1.00
0.59
1.06
1.27
0.88
1.58
1.11
2.17
0.80
-0.56
0.66
0.73
2.33
0.28
1.38
1.34
-0.25
2.29
0.32
1.18
-2.73
1.84
-1.35
0.74];
x4=[1.69
2.31
1.25
1.77
1.01
0.42
2.05
1.21
0.65
0.53
0.47
1.21
1.91
0.13
1.47
2.56
2.25
1.52
2.39
0.52
1.20
1.70
1.46
2.09
1.99
1.31
1.16
1.51
2.52
2.05
1.42
0.78
0.64
3.17
2.62
2.18
2.41
1.51
2.17
2.45
1.41
1.03
0.28
2.73
2.64
0.62
1.82
1.70
0.49
0.72
1.22];
y=1:1:51;
x1max=max(x1);
y1max=find(x1==x1max);
x2max=max(x2);
y2max=find(x2==x2max);
x4max=max(x4);
y4max=find(x4==x4max);
figure
plot(y,x1,'-o','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(y,x2,'-ks','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y,x4,'-P','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y1max,x1max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y2max,x2max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y4max,x4max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
legend('MNIST-1','MNIST-2','MNIST-4','location','best')
xlabel('Trial','fontweight','bold','FontSize',16)
set(gca,'XTickLabelMode','auto')
ylabel('Gain(%)','fontweight','bold','FontSize',16)
set(gca,'XLim',[1 51])
ax = gca;
ax.FontSize = 16; 
fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig, 'MNIST_3.pdf', '-dpdf')