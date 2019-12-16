x1=[0
    -0.57
    -0.57
    1.65
    2.49
    3.41
    3.41];
x2=[0
    2.03
    2.74
    2.74
    2.74
    2.74
    2.74];
x4=[0
    2.40
    2.40
    2.40
    2.40
    2.40
    2.40];
y=0:10:60;
x1max=max(x1);
y1max=find(x1==x1max);
y1max=0+(y1max-1)*10;
x2max=max(x2);
y2max=find(x2==x2max);
y2max=0+(y2max-1)*10;
x4max=max(x4);
y4max=find(x4==x4max);
y4max=0+(y4max-1)*10;
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
xlabel('Minutes','fontweight','bold','FontSize',16)
set(gca,'XTickLabelMode','auto')
ylabel('Max Gain(%)','fontweight','bold','FontSize',16)
set(gca,'XLim',[0 60])
%set(leg,'location','best')
ax = gca;
ax.FontSize = 16; 
fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig, 'MNIST_time.pdf', '-dpdf')